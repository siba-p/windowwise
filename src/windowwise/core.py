"""Planning, WHAM, bootstrap errors and statistical-inefficiency estimates for 1D umbrella sampling.

Bias convention (GROMACS pull code, Grossfield WHAM): U_i(x) = 0.5 * k_i * (x - c_i)**2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

KB = {"kJ/mol": 0.0083144626, "kcal/mol": 0.0019872043}


def kT(T: float, unit: str = "kJ/mol") -> float:
    """Thermal energy k_B T in the requested energy unit."""
    try:
        return KB[unit] * T
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unit must be one of {list(KB)}") from exc


def gauss_overlap(d: float, sigma: float) -> float:
    """Overlap coefficient of two equal-width Gaussians whose centres are d apart."""
    return 2.0 * NormalDist().cdf(-d / (2.0 * sigma))


@dataclass
class Window:
    """One umbrella window: its sampled coordinate, restraint centre and force constant."""

    data: np.ndarray
    center: float
    k: float
    name: str = ""

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)


@dataclass
class Plan:
    k: float
    spacing: float
    sigma: float
    centers: np.ndarray
    overlap: float
    shift: float
    rule: str

    @property
    def n(self) -> int:
        return len(self.centers)


def plan(xmin: float, xmax: float, T: float = 300.0, unit: str = "kJ/mol", overlap: float = 0.3,
         slope: float | None = None, k: float | None = None, spacing: float | None = None,
         k_min: float = 0.0) -> Plan:
    """Design umbrella windows.

    Exactly one rule applies, in this order: a fixed ``spacing`` (k follows from the target overlap),
    a fixed ``k`` (spacing follows), or the steepest expected PMF ``slope`` (k is chosen so that the
    slope shifts a window by at most half a spacing; never below ``k_min``).
    """
    if not xmax > xmin:
        raise ValueError("xmax must be larger than xmin")
    if not 0 < overlap < 1:
        raise ValueError("overlap must be between 0 and 1")
    kt = kT(T, unit)
    z = -NormalDist().inv_cdf(overlap / 2.0)  # spacing = 2 z sigma
    if spacing is not None:
        d = float(spacing)
        kk = kt * (2 * z / d) ** 2
        rule = "k from spacing and target overlap"
    elif k is not None:
        kk = float(k)
        d = 2 * z * math.sqrt(kt / kk)
        rule = "spacing from k and target overlap"
    else:
        k_slope = (abs(slope) / z) ** 2 / kt if slope else 0.0
        kk = max(k_min, k_slope)
        if kk <= 0:
            raise ValueError("give slope, k, spacing or a positive k_min")
        d = 2 * z * math.sqrt(kt / kk)
        rule = "k set by the steepest expected slope" if k_slope > k_min else "k set by the minimum force constant"
    sigma = math.sqrt(kt / kk)
    n = max(2, math.ceil((xmax - xmin) / d - 1e-9) + 1)
    centers = np.linspace(xmin, xmax, n)
    step = centers[1] - centers[0]
    return Plan(k=kk, spacing=step, sigma=sigma, centers=centers, overlap=gauss_overlap(step, sigma),
                shift=(abs(slope) / kk) if slope else 0.0, rule=rule)


def statistical_inefficiency(x: np.ndarray, max_lag: int = 600, target_points: int = 4000) -> float:
    """g = 1 + 2 sum_t (1 - t/N) rho(t), summed until rho first drops to zero (>= 1)."""
    x = np.asarray(x, dtype=float)
    stride = max(1, len(x) // target_points)
    y = x[::stride]
    n = len(y)
    if n < 20:
        return 1.0
    y = y - y.mean()
    var = np.dot(y, y) / n
    if var == 0:
        return 1.0
    g = 1.0
    for t in range(1, min(n // 4, max_lag)):
        rho = np.dot(y[:-t], y[t:]) / ((n - t) * var)
        if rho <= 0:
            break
        g += 2 * rho * (1 - t / n)
    return max(1.0, g * stride)


@dataclass
class Bins:
    lo: float
    hi: float
    nbins: int

    @property
    def width(self) -> float:
        return (self.hi - self.lo) / self.nbins

    @property
    def centers(self) -> np.ndarray:
        return self.lo + (np.arange(self.nbins) + 0.5) * self.width

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.lo, self.hi, self.nbins + 1)

    @classmethod
    def covering(cls, windows: list[Window], nbins: int) -> "Bins":
        lo = min(float(w.data.min()) for w in windows)
        hi = max(float(w.data.max()) for w in windows)
        pad = (hi - lo) * 1e-6
        return cls(lo - pad, hi + pad, nbins)


def histograms(windows: list[Window], bins: Bins) -> np.ndarray:
    return np.array([np.histogram(w.data, bins=bins.edges)[0] for w in windows], dtype=float)


@dataclass
class WhamResult:
    F: np.ndarray          # PMF per bin, zero at the minimum, NaN where unsampled
    f: np.ndarray          # dimensionless window free energies
    iterations: int
    converged: bool


def wham(hists: np.ndarray, centers, ks, bins: Bins, kt: float, tol: float = 1e-7,
         max_iter: int = 100000, f0: np.ndarray | None = None) -> WhamResult:
    """Self-consistent 1D WHAM (Kumar et al. 1992). Linear-space iteration with a log-space fallback."""
    centers = np.asarray(centers, float)
    ks = np.asarray(ks, float)
    x = bins.centers
    bias = 0.5 * ks[:, None] * (x[None, :] - centers[:, None]) ** 2 / kt   # (W, B), in kT
    N = hists.sum(axis=1)
    num = hists.sum(axis=0)
    sampled = num > 0
    f = np.zeros(len(centers)) if f0 is None else np.array(f0, float)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        E = np.exp(-bias)
        ok, it, delta = True, 0, np.inf
        while it < max_iter and delta > tol:
            denom = (N * np.exp(f)) @ E
            P = np.where(sampled, num / denom, 0.0)
            f_new = -np.log(E @ P)
            f_new -= f_new[0]
            if not np.all(np.isfinite(f_new)):
                ok = False
                break
            delta = np.max(np.abs(f_new - f))
            f = f_new
            it += 1
        if ok:
            lnP = np.where(sampled, np.log(P), -np.inf)
            ok = np.all(np.isfinite(lnP[sampled]))
    if not ok:  # log-space fallback for very deep profiles
        f = np.zeros(len(centers)) if f0 is None else np.array(f0, float)
        lnN = np.log(np.maximum(N, 1e-300))
        lnnum = np.log(np.where(sampled, num, 1.0))
        it, delta = 0, np.inf
        while it < max_iter and delta > tol:
            a = (lnN + f)[:, None] - bias
            m = a.max(axis=0)
            lnden = m + np.log(np.exp(a - m).sum(axis=0))
            lnP = np.where(sampled, lnnum - lnden, -np.inf)
            b = lnP[None, :] - bias
            mb = b.max(axis=1)
            f_new = -(mb + np.log(np.exp(b - mb[:, None]).sum(axis=1)))
            f_new -= f_new[0]
            delta = np.max(np.abs(f_new - f))
            f = f_new
            it += 1
    F = np.where(sampled, -kt * lnP, np.nan)
    F -= np.nanmin(F)
    return WhamResult(F=F, f=f, iterations=it, converged=bool(delta <= tol))


def bootstrap_errors(windows: list[Window], bins: Bins, kt: float, f0: np.ndarray, nboot: int = 20,
                     g: list[float] | None = None, seed: int = 11) -> np.ndarray:
    """Block-bootstrap standard error of the PMF; blocks of ~2g frames keep time correlation."""
    rng = np.random.default_rng(seed)
    g = g or [statistical_inefficiency(w.data) for w in windows]
    centers = [w.center for w in windows]
    ks = [w.k for w in windows]
    samples = []
    for _ in range(nboot):
        hs = []
        for w, gi in zip(windows, g):
            n = len(w.data)
            L = int(max(1, min(n, round(2 * gi))))
            nblocks = math.ceil(n / L)
            starts = rng.integers(0, max(1, n - L + 1), size=nblocks)
            idx = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
            hs.append(np.histogram(w.data[idx], bins=bins.edges)[0])
        res = wham(np.array(hs, float), centers, ks, bins, kt, tol=1e-5, max_iter=40000, f0=f0)
        samples.append(res.F)
    S = np.array(samples)
    with np.errstate(invalid="ignore"):
        err = np.nanstd(S, axis=0)
    err[np.sum(np.isfinite(S), axis=0) < 2] = np.nan
    return err
