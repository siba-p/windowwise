"""Gaussian-process umbrella integration with active-learning window placement.

Each window gives an exact estimate of the biased average of the mean force,
    <dA/dxi>_i = k_i (c_i - <xi>_i),
with variance k_i^2 * var(xi_i) * g_i / N_i (g = statistical inefficiency). Treating these as noisy
derivative observations of a Gaussian-process prior on A(xi) (squared-exponential kernel) gives a PMF
with a Bayesian uncertainty band (cf. Stecher, Bernstein & Csanyi, JCTC 10, 4079, 2014). The posterior
variance of the mean force then tells us where a new window is most informative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Window, statistical_inefficiency


@dataclass
class GPResult:
    x: np.ndarray            # grid
    A: np.ndarray            # posterior mean PMF (zero at its minimum)
    A_std: np.ndarray        # posterior std of A(x) - A(x_min)
    dA: np.ndarray           # posterior mean force
    dA_std: np.ndarray       # posterior std of the mean force
    obs_x: np.ndarray
    obs_y: np.ndarray
    obs_sd: np.ndarray
    lengthscale: float
    amplitude: float
    log_marginal: float


def mean_force_observations(windows: list[Window]):
    """(x, y, sd): window means, mean-force estimates and their standard errors."""
    x, y, sd = [], [], []
    for w in windows:
        m = float(w.data.mean())
        var = float(w.data.var(ddof=1))
        g = statistical_inefficiency(w.data)
        x.append(m)
        y.append(w.k * (w.center - m))
        sd.append(w.k * np.sqrt(var * g / len(w.data)))
    return np.array(x), np.array(y), np.array(sd)


def _kernels(xa, xb, ell, s2):
    d = xa[:, None] - xb[None, :]
    e = np.exp(-0.5 * d ** 2 / ell ** 2)
    k_ff = s2 * e                                   # cov(A(a), A(b))
    k_fd = s2 * e * d / ell ** 2                    # cov(A(a), A'(b))
    k_dd = s2 * e * (1.0 / ell ** 2 - d ** 2 / ell ** 4)  # cov(A'(a), A'(b))
    return k_ff, k_fd, k_dd


def _fit(x, y, sd, ell, s2):
    _, _, Kdd = _kernels(x, x, ell, s2)
    K = Kdd + np.diag(sd ** 2 + 1e-10 * s2 / ell ** 2)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return None
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(x) * np.log(2 * np.pi)
    return L, alpha, lml


def gp_pmf(windows: list[Window], grid: np.ndarray | None = None, npoints: int = 300,
           lengthscale: float | None = None, amplitude: float | None = None) -> GPResult:
    """Fit the GP to window mean forces; hyperparameters by maximum marginal likelihood (grid search)."""
    x, y, sd = mean_force_observations(windows)
    order = np.argsort(x)
    x, y, sd = x[order], y[order], sd[order]
    span = x.max() - x.min()
    if grid is None:
        grid = np.linspace(x.min(), x.max(), npoints)
    spacing = np.median(np.diff(x)) if len(x) > 1 else span
    ells = [lengthscale] if lengthscale else np.geomspace(max(spacing, span / 200), span / 1.5, 28)
    yscale = max(np.std(y), np.max(np.abs(y)) / 3, 1e-9)
    amps = [amplitude] if amplitude else None
    best = None
    for ell in ells:
        # amplitude of A ~ (mean-force scale) * lengthscale
        cand = amps or (yscale * ell * np.geomspace(0.1, 10, 15))
        for a in cand:
            fit = _fit(x, y, sd, ell, a ** 2)
            if fit and (best is None or fit[2] > best[0]):
                best = (fit[2], ell, a, fit)
    if best is None:
        raise RuntimeError("GP fit failed")
    lml, ell, a, (L, alpha, _) = best
    s2 = a ** 2
    _, kfd_gx, kdd_gx = _kernels(grid, x, ell, s2)
    kff_gg = _kernels(grid, grid, ell, s2)[0]
    A = kfd_gx @ alpha
    dA = kdd_gx @ alpha
    V = np.linalg.solve(L, kfd_gx.T)            # for cov of A
    Vd = np.linalg.solve(L, kdd_gx.T)           # for var of A'
    covA = kff_gg - V.T @ V
    imin = int(np.argmin(A))
    varrel = np.diag(covA) + covA[imin, imin] - 2 * covA[:, imin]
    vard = s2 / ell ** 2 - np.sum(Vd ** 2, axis=0)
    A = A - A[imin]
    return GPResult(x=grid, A=A, A_std=np.sqrt(np.maximum(varrel, 0)), dA=dA, dA_std=np.sqrt(np.maximum(vard, 0)),
                    obs_x=x, obs_y=y, obs_sd=sd, lengthscale=ell, amplitude=a, log_marginal=float(lml))


def suggest_windows(windows: list[Window], n: int = 3, k: float | None = None, xmin: float | None = None,
                    xmax: float | None = None, min_separation: float | None = None, gp: GPResult | None = None):
    """Active learning: place new windows where the posterior mean-force uncertainty is largest.

    Uses a 'kriging believer' loop: after each pick a pseudo-observation with the typical noise is added,
    so successive picks spread out. Returns dicts with the target position, the restraint centre that
    lands the window there (offset by A'/k) and the force constant.
    """
    gp = gp or gp_pmf(windows)
    x = list(gp.obs_x)
    y = list(gp.obs_y)
    sd = list(gp.obs_sd)
    lo = gp.x.min() if xmin is None else xmin
    hi = gp.x.max() if xmax is None else xmax
    grid = np.linspace(lo, hi, 400)
    kk = k or float(np.median([w.k for w in windows]))
    sep = min_separation or 0.5 * float(np.median(np.diff(np.sort(gp.obs_x))))
    typical_sd = float(np.median(gp.obs_sd))
    ell, s2 = gp.lengthscale, gp.amplitude ** 2
    picks = []
    for _ in range(n):
        xa, ya, sa = np.array(x), np.array(y), np.array(sd)
        fit = _fit(xa, ya, sa, ell, s2)
        L, alpha, _ = fit
        _, _, kdd = _kernels(grid, xa, ell, s2)
        Vd = np.linalg.solve(L, kdd.T)
        std = np.sqrt(np.maximum(s2 / ell ** 2 - np.sum(Vd ** 2, axis=0), 0))
        dmin = np.min(np.abs(grid[:, None] - xa[None, :]), axis=1)
        score = np.where(dmin >= sep, std, -np.inf)
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        target = float(grid[j])
        force = float(kdd[j] @ alpha)
        picks.append({"target": target, "center": target + force / kk, "k": kk,
                      "mean_force": force, "mean_force_std": float(std[j])})
        x.append(target)
        y.append(force)
        sd.append(typical_sd)
    return picks
