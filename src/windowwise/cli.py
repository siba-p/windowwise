"""Command-line interface: ``windowwise plan|analyze|gp|demo``."""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np

from . import __version__
from .analysis import analyze, synthetic
from .core import plan
from .gp import gp_pmf, suggest_windows
from .io import load_windows, mdp_snippet


def _files(patterns):
    out = []
    for p in patterns or []:
        out.extend(sorted(glob.glob(p)) or [p])
    return out


def _common(p):
    p.add_argument("files", nargs="*", help="pullx.xvg (or 2-column) files, one per window; globs allowed")
    p.add_argument("-m", "--metadata", help="Grossfield-style metadata: 'path center k' per line")
    p.add_argument("-k", type=float, help="force constant for all windows (if no metadata)")
    p.add_argument("--centers", type=float, nargs="+", help="window centres (default: read from file names)")
    p.add_argument("--column", type=int, default=1, help="data column (0 = time; default 1)")
    p.add_argument("-T", "--temperature", type=float, default=300.0)
    p.add_argument("--unit", default="kJ/mol", choices=["kJ/mol", "kcal/mol"])
    p.add_argument("--length-unit", default="nm")
    p.add_argument("-o", "--output", help="output prefix (writes <prefix>_pmf.csv, <prefix>_repair.mdp, <prefix>.png)")


def _plot(path, x, F, err, title, extra=None, xlabel="xi", ylabel="F"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
    if err is not None:
        ax.fill_between(x, F - err, F + err, color="#2a78d6", alpha=0.2, lw=0)
    ax.plot(x, F, color="#2a78d6", lw=2, label=title)
    if extra:
        for xs, ys, kw in extra:
            ax.plot(xs, ys, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def cmd_plan(a):
    p = plan(a.xmin, a.xmax, a.temperature, a.unit, a.overlap / 100, slope=a.slope, k=a.k, spacing=a.spacing, k_min=a.k_min)
    L, E = a.length_unit, a.unit
    print(f"rule              : {p.rule}")
    print(f"force constant k  : {p.k:.0f} {E}/{L}^2")
    print(f"spacing           : {p.spacing:.4f} {L}")
    print(f"windows           : {p.n}")
    print(f"window width sigma: {p.sigma:.4f} {L}")
    print(f"neighbour overlap : {p.overlap * 100:.1f}%")
    if a.slope:
        print(f"max slope shift   : {p.shift:.4f} {L} ({p.shift / p.spacing * 100:.0f}% of spacing)")
    snippet = mdp_snippet(p.centers, [p.k] * p.n, a.geometry)
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(snippet)
        print(f"wrote {a.output}")
    else:
        print()
        print(snippet)


def cmd_analyze(a):
    ws = load_windows(_files(a.files), a.metadata, a.k, a.centers, a.column)
    res = analyze(ws, a.temperature, a.unit, a.bins, a.bootstrap, a.overlap_min / 100, a.length_unit)
    print(f"{len(ws)} windows · WHAM {'converged' if res.converged else 'NOT converged'} in {res.iterations} iterations")
    if not res.flags:
        print("No issues found.")
    for f in res.flags:
        print(f"[{f.level.upper():8s}] {f.text}")
    if res.suggestions:
        print("\nSuggested repair windows:")
        for s in res.suggestions:
            print(f"  {s.reason:9s} target {s.target:.4f} -> pull-coord1-init = {s.center:.4f}, pull-coord1-k = {round(s.k)}")
    if a.output:
        with open(a.output + "_pmf.csv", "w") as fh:
            fh.write(f"# WindowWise PMF, T={a.temperature} K, {a.unit}, zero at minimum\nxi,F,F_err\n")
            for x, F, e in zip(res.x, res.F, res.err if res.err is not None else [np.nan] * len(res.x)):
                fh.write(f"{x:.6f},{F:.5f},{e:.5f}\n")
        print(f"wrote {a.output}_pmf.csv")
        if res.suggestions:
            with open(a.output + "_repair.mdp", "w") as fh:
                fh.write(mdp_snippet([s.center for s in res.suggestions], [s.k for s in res.suggestions], header="repair windows"))
            print(f"wrote {a.output}_repair.mdp")
        _plot(a.output + ".png", res.x, res.F, res.err, "WHAM ± bootstrap", xlabel=f"ξ ({a.length_unit})", ylabel=f"F ({a.unit})")
    if a.json:
        json.dump({"x": res.x.tolist(), "F": np.nan_to_num(res.F, nan=-1).tolist(),
                   "flags": [f.__dict__ for f in res.flags], "suggestions": [s.__dict__ for s in res.suggestions]},
                  sys.stdout, indent=1)


def cmd_gp(a):
    ws = load_windows(_files(a.files), a.metadata, a.k, a.centers, a.column)
    g = gp_pmf(ws)
    print(f"GP umbrella integration: {len(ws)} windows, lengthscale {g.lengthscale:.4f} {a.length_unit}, "
          f"log marginal likelihood {g.log_marginal:.2f}")
    picks = suggest_windows(ws, n=a.n, gp=g)
    print(f"\nMost informative next {len(picks)} windows (active learning, max posterior mean-force uncertainty):")
    for p in picks:
        print(f"  target {p['target']:.4f} -> pull-coord1-init = {p['center']:.4f}, pull-coord1-k = {round(p['k'])} "
              f"(mean-force sd {p['mean_force_std']:.1f})")
    if a.output:
        with open(a.output + "_gp.csv", "w") as fh:
            fh.write("xi,A,A_std,dA,dA_std\n")
            for row in zip(g.x, g.A, g.A_std, g.dA, g.dA_std):
                fh.write(",".join(f"{v:.6f}" for v in row) + "\n")
        print(f"wrote {a.output}_gp.csv")
        _plot(a.output + "_gp.png", g.x, g.A, 1.96 * g.A_std, "GP umbrella integration (95%)",
              xlabel=f"ξ ({a.length_unit})", ylabel=f"A ({a.unit})")


def cmd_demo(a):
    import math
    D, alpha, x0 = 30.0, 3.5, 0.7

    def F(x):
        return D * ((1 - math.exp(-alpha * (x - x0))) ** 2 - 1)
    centers = [round(c, 2) for c in np.arange(0.5, 2.501, 0.1) if round(c, 2) not in (1.4, 1.5)]
    ws = synthetic(F, centers, 1000.0, n=3000, seed=4)
    res = analyze(ws, 300, "kJ/mol", 120, 20)
    print("Demo: Morse-type adsorption PMF (30 kJ/mol well), soft springs on the wall and two missing windows.\n")
    for f in res.flags:
        print(f"[{f.level.upper():8s}] {f.text}")
    print("\nRepair windows:")
    for s in res.suggestions:
        print(f"  {s.reason:9s} pull-coord1-init = {s.center:.4f}, pull-coord1-k = {round(s.k)}")
    picks = suggest_windows(ws, n=3)
    print("\nGP active learning, next 3 windows:", ", ".join(f"{p['target']:.3f}" for p in picks))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="windowwise", description="Plan, check and repair umbrella sampling.")
    ap.add_argument("--version", action="version", version=f"windowwise {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="design window spacing and force constant")
    p.add_argument("xmin", type=float)
    p.add_argument("xmax", type=float)
    p.add_argument("-T", "--temperature", type=float, default=300.0)
    p.add_argument("--unit", default="kJ/mol", choices=["kJ/mol", "kcal/mol"])
    p.add_argument("--length-unit", default="nm")
    p.add_argument("--overlap", type=float, default=30, help="target neighbour overlap in percent (default 30)")
    p.add_argument("--slope", type=float, help="steepest expected |dF/dxi| (energy/length)")
    p.add_argument("-k", type=float, help="fixed force constant")
    p.add_argument("--spacing", type=float, help="fixed spacing")
    p.add_argument("--k-min", type=float, default=1000.0)
    p.add_argument("--geometry", default="distance")
    p.add_argument("-o", "--output", help="write the .mdp snippet to this file")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("analyze", help="WHAM + bootstrap + diagnostics + repair windows")
    _common(p)
    p.add_argument("--bins", type=int, default=120)
    p.add_argument("--bootstrap", type=int, default=20)
    p.add_argument("--overlap-min", type=float, default=10, help="overlap threshold in percent (default 10)")
    p.add_argument("--json", action="store_true", help="print machine-readable results")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("gp", help="Gaussian-process umbrella integration + active-learning window suggestions")
    _common(p)
    p.add_argument("-n", type=int, default=3, help="number of new windows to propose")
    p.set_defaults(func=cmd_gp)

    p = sub.add_parser("demo", help="run the built-in demonstration")
    p.set_defaults(func=cmd_demo)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":  # pragma: no cover
    main()
