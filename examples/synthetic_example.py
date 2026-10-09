"""End-to-end example on synthetic data with a known answer.

Run:  python examples/synthetic_example.py
"""
import math

import numpy as np

import windowwise as ww


def pmf(x):  # Morse-type adsorption profile, 30 kJ/mol well at 0.7 nm
    return 30 * ((1 - math.exp(-3.5 * (x - 0.7))) ** 2 - 1)


# 1. plan
p = ww.plan(0.5, 2.5, T=300, overlap=0.3, slope=120, k_min=1000)
print(f"plan: k = {p.k:.0f} kJ/mol/nm^2, {p.n} windows, spacing {p.spacing:.3f} nm")

# 2. simulate a deliberately poor set (soft springs, missing windows) and analyse it
centers = [c for c in np.round(np.arange(0.5, 2.501, 0.1), 2) if c not in (1.4, 1.5)]
windows = ww.synthetic(pmf, centers, 1000.0, n=3000, seed=4)
res = ww.analyze(windows, T=300, nboot=20)
for f in res.flags:
    print(f"[{f.level}] {f.text}")
print(ww.mdp_snippet([s.center for s in res.suggestions], [s.k for s in res.suggestions], header="repair windows"))

# 3. Bayesian PMF and active learning
gp = ww.gp_pmf(windows)
for s in ww.suggest_windows(windows, n=3, gp=gp):
    print(f"next window: target {s['target']:.3f} nm -> centre {s['center']:.3f} nm, k {s['k']:.0f}")
