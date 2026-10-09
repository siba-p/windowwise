"""Full analysis: WHAM + bootstrap + overlap diagnostics + repair-window suggestions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

from .core import Bins, Window, bootstrap_errors, histograms, kT, statistical_inefficiency, wham


@dataclass
class Flag:
    level: str      # "warning" | "critical"
    kind: str       # "overlap" | "drift" | "samples" | "wham"
    text: str
    x: float = float("nan")


@dataclass
class Suggestion:
    center: float
    k: float
    target: float
    reason: str     # "fill gap" | "stiffen"
    slope: float


@dataclass
class Analysis:
    bins: Bins
    F: np.ndarray
    err: np.ndarray | None
    windows: list[Window]
    means: np.ndarray
    g: np.ndarray
    pairs: list[tuple[int, int, float]]
    flags: list[Flag] = field(default_factory=list)
    suggestions: list[Suggestion] = field(default_factory=list)
    converged: bool = True
    iterations: int = 0

    @property
    def x(self) -> np.ndarray:
        return self.bins.centers


def _slope_at(F, bins: Bins, x0: float) -> float:
    b = int(np.clip(np.floor((x0 - bins.lo) / bins.width), 0, bins.nbins - 1))
    lo = hi = b
    while lo > 0 and b - lo < 4 and np.isfinite(F[lo - 1]):
        lo -= 1
    while hi < bins.nbins - 1 and hi - b < 4 and np.isfinite(F[hi + 1]):
        hi += 1
    if hi == lo or not (np.isfinite(F[lo]) and np.isfinite(F[hi])):
        return float("nan")
    return float((F[hi] - F[lo]) / ((hi - lo) * bins.width))


def analyze(windows: list[Window], T: float = 300.0, unit: str = "kJ/mol", nbins: int = 120, nboot: int = 20,
            overlap_min: float = 0.1, length_unit: str = "nm") -> Analysis:
    windows = [w for w in windows if len(w.data) > 1]
    if len(windows) < 2:
        raise ValueError("need at least two windows with data")
    kt = kT(T, unit)
    L, E = length_unit, unit
    bins = Bins.covering(windows, nbins)
    H = histograms(windows, bins)
    res = wham(H, [w.center for w in windows], [w.k for w in windows], bins, kt)
    g = np.array([statistical_inefficiency(w.data) for w in windows])
    err = bootstrap_errors(windows, bins, kt, res.f, nboot, list(g)) if nboot > 0 else None
    means = np.array([w.data.mean() for w in windows])
    order = np.argsort([w.center for w in windows])
    pairs = []
    for a, b in zip(order[:-1], order[1:]):
        pa, pb = H[a] / max(H[a].sum(), 1), H[b] / max(H[b].sum(), 1)
        pairs.append((int(a), int(b), float(np.minimum(pa, pb).sum())))
    spacings = [abs(windows[b].center - windows[a].center) for a, b, _ in pairs]
    med = float(np.median(spacings)) if spacings else 0.0

    out = Analysis(bins=bins, F=res.F, err=err, windows=windows, means=means, g=g, pairs=pairs,
                   converged=res.converged, iterations=res.iterations)
    z = -NormalDist().inv_cdf(max(overlap_min, 0.2) / 2)
    for a, b, ov in pairs:
        if ov >= overlap_min:
            continue
        wa, wb = windows[a], windows[b]
        xa, xb = means[a], means[b]
        gap = abs(xb - xa)
        out.flags.append(Flag("critical" if ov < overlap_min / 3 else "warning", "overlap",
                              f"Low overlap ({ov * 100:.1f}%) between windows at {wa.center:.3f} and {wb.center:.3f} {L}.",
                              (xa + xb) / 2))
        s = _slope_at(res.F, bins, (xa + xb) / 2)
        if not np.isfinite(s):
            s = 0.5 * (wa.k * (wa.center - xa) + wb.k * (wb.center - xb))
        k_new = max(wa.k, wb.k, 4 * abs(s) / max(gap, 1e-12))
        sig = math.sqrt(kt / k_new)
        n_add = max(1, math.ceil(gap / (2 * z * sig)) - 1)
        for q in range(1, n_add + 1):
            target = xa + (xb - xa) * q / (n_add + 1)
            out.suggestions.append(Suggestion(target + s / k_new, k_new, target, "fill gap", s))
    for i, w in enumerate(windows):
        drift = abs(means[i] - w.center)
        if med > 0 and drift > 0.5 * med:
            s_shift = w.k * (w.center - means[i])
            s_pmf = _slope_at(res.F, bins, w.center)
            s = s_pmf if np.isfinite(s_pmf) and abs(s_pmf) > abs(s_shift) else s_shift
            out.flags.append(Flag("warning", "drift",
                                  f"Window at {w.center:.3f} {L} sits {drift:.3f} {L} from its centre: the spring is too soft "
                                  f"for the local slope (~{abs(s):.0f} {E}/{L}).", float(means[i])))
            out.suggestions.append(Suggestion(w.center, max(w.k, math.ceil(2 * abs(s) / med)), w.center, "stiffen", s))
        neff = len(w.data) / g[i]
        if neff < 50:
            out.flags.append(Flag("warning", "samples",
                                  f"Window at {w.center:.3f} {L} has only ~{neff:.0f} independent samples "
                                  f"(statistical inefficiency {g[i]:.1f}).", float(means[i])))
    if not res.converged:
        out.flags.append(Flag("critical", "wham", "WHAM did not converge; check centres and force constants."))
    out.suggestions.sort(key=lambda s: s.target)
    return out


def synthetic(F, centers, k, T=300.0, unit="kJ/mol", n=3000, stride=5, burn=2000, seed=1, names=True):
    """Metropolis Monte Carlo samples of a known PMF F(x) under harmonic biases (for demos and tests)."""
    rng = np.random.default_rng(seed)
    kt = kT(T, unit)
    ks = np.broadcast_to(np.asarray(k, float), (len(centers),))
    out = []
    for i, (c, kk) in enumerate(zip(centers, ks)):
        def energy(x):
            return F(x) + 0.5 * kk * (x - c) ** 2
        x, e = c, energy(c)
        step = math.sqrt(kt / kk)
        total = burn + n * stride
        props = rng.uniform(-step, step, total)
        us = rng.random(total)
        data = np.empty(n)
        for s in range(total):
            y = x + props[s]
            ey = energy(y)
            if ey <= e or us[s] < math.exp(-(ey - e) / kt):
                x, e = y, ey
            if s >= burn and (s - burn) % stride == 0:
                data[(s - burn) // stride] = x
        out.append(Window(data, float(c), float(kk), f"window_{i:02d}_{c:.3f}.xvg" if names else ""))
    return out
