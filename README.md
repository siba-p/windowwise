# WindowWise

**Plan, check and repair umbrella-sampling simulations.**

[![tests](https://github.com/siba-p/windowwise/actions/workflows/ci.yml/badge.svg)](https://github.com/siba-p/windowwise/actions/workflows/ci.yml)
[![web app](https://img.shields.io/badge/web%20app-siba--p.github.io%2Fwindowwise-3b50e0)](https://siba-p.github.io/windowwise/)
![python](https://img.shields.io/badge/python-3.9%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Umbrella sampling fails quietly. Windows spaced too far apart, or springs too soft for a steep wall, leave holes in the free-energy profile that WHAM smooths over. You usually find out only after the compute is spent. WindowWise covers the whole loop:

| | |
|---|---|
| **Plan** | Force constant and spacing from first principles, for a target neighbour overlap and the steepest expected slope. Writes GROMACS pull settings. |
| **Analyse** | WHAM on `pullx.xvg` files (or Grossfield metadata), with block-bootstrap errors whose block length is set by each window's statistical inefficiency. |
| **Diagnose** | Flags gaps, windows that drift off their centre, under-sampled windows and non-convergence. |
| **Repair** | Proposes exactly which windows to add or re-run. New centres are offset by F′/k so they land in the gap, and k is raised where the PMF is steep. Output is a ready-to-paste `.mdp` snippet. |
| **Learn (ML)** | **Gaussian-process umbrella integration**: a PMF with a Bayesian uncertainty band from window mean forces, plus **active learning** that proposes the most informative next windows. |

Use it **in the browser** (no install; files never leave your computer): **https://siba-p.github.io/windowwise/**, or install the Python package below for scripts and HPC workflows.

## Install

```bash
pip install "git+https://github.com/siba-p/windowwise.git#egg=windowwise[plot]"
```

The only dependency is NumPy; matplotlib is optional for plots.

## Quick start

```bash
# 1. design windows before simulating
windowwise plan 0.5 2.5 --slope 80 --overlap 30 -T 300 -o windows.mdp

# 2. analyse finished windows (GROMACS pullx files, centres read from file names)
windowwise analyze pullx_*.xvg -k 1000 -T 300 -o run1
#    ...or with a Grossfield-style metadata file: "path centre k" per line
windowwise analyze -m metadata.dat -o run1          # writes run1_pmf.csv, run1_repair.mdp, run1.png

# 3. Bayesian PMF + where to put the next windows
windowwise gp -m metadata.dat -n 3 -o run1

windowwise demo                                       # built-in example with deliberate mistakes
```

Python API:

```python
import windowwise as ww

windows = ww.load_windows(metadata="metadata.dat")
res = ww.analyze(windows, T=300, nboot=20)
for flag in res.flags:
    print(flag.level, flag.text)
print(ww.mdp_snippet([s.center for s in res.suggestions], [s.k for s in res.suggestions]))

gp = ww.gp_pmf(windows)                 # posterior mean / std of A(xi) and of the mean force
next_windows = ww.suggest_windows(windows, n=3, gp=gp)
```

## Method

- **Planning.** On a locally flat PMF a window samples a Gaussian of width σ = √(k<sub>B</sub>T/k). Two neighbours a distance d apart overlap by 2Φ(−d/2σ). A slope F′ shifts every window by F′/k, so k is chosen to keep that shift below half a spacing: k ≥ (F′/z)²/k<sub>B</sub>T, with z = −Φ⁻¹(overlap/2).
- **WHAM.** Self-consistent 1D WHAM (Kumar *et al.*, *J. Comput. Chem.* **13**, 1011, 1992), iterated in linear space with precomputed Boltzmann factors and an automatic log-space fallback for very deep profiles. Bias U = ½k(ξ − ξ₀)², as in the GROMACS pull code and Grossfield's WHAM.
- **Errors.** Block bootstrap of each time series. The block length is 2g, where g is the statistical inefficiency from the integrated autocorrelation function.
- **Repair.** Gaps below the overlap threshold receive new windows spaced for the target overlap. Their k satisfies |F′|/k ≤ gap/4, with the local F′ taken from the PMF or, where unsampled, from the spring balance k(c − ⟨ξ⟩). Drifting windows get k ≥ 2|F′|/spacing.
- **GP umbrella integration.** Each window gives an exact estimate of its biased-average mean force, ⟨A′⟩ᵢ = kᵢ(cᵢ − ⟨ξ⟩ᵢ), with variance kᵢ² var(ξ) gᵢ/Nᵢ. These are derivative observations of a squared-exponential GP prior on A(ξ), with hyperparameters set by maximum marginal likelihood (cf. Stecher, Bernstein & Csányi, *JCTC* **10**, 4079, 2014). Active learning picks new windows where the posterior mean-force uncertainty is largest, using a kriging-believer loop so successive picks spread out.

## Validation

All numbers are from `pytest`, against an analytic Morse-type adsorption PMF (30 kJ/mol well, Metropolis-sampled windows at 300 K):

| test | result |
|---|---|
| WHAM vs exact PMF, 41 windows | RMSE **0.33 kJ/mol**; median bootstrap error 0.35 kJ/mol |
| GP umbrella integration, 41 / 21 / 11 windows | RMSE 0.36 / 0.52 / 1.05 kJ/mol |
| GP 95% band covering the exact PMF (41 / 11 windows) | 83% / 95%. The band is mildly over-confident with dense windows, because ⟨A′⟩ differs from A′(⟨ξ⟩) at second order. |
| Removed block of windows | detected; repair windows and GP active-learning picks land inside the gap |
| Soft springs on a steep wall | flagged as drift; slope estimate within 2% of the analytic value |
| ~800 k<sub>B</sub>T-deep PMF | converges via the log-space fallback |

Run them yourself with `pip install -e ".[test]" && pytest`.

## Limitations

One-dimensional, non-periodic coordinates; harmonic restraints; centres must be known (from metadata, file names or `--centers`).

## Citation

If WindowWise helps your work, please cite it (see `CITATION.cff`):

> S. Panigrahy, *WindowWise: planning, diagnostics and repair for umbrella sampling*, 2026. https://github.com/siba-p/windowwise

## License

MIT © Sibasankar Panigrahy
