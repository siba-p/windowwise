import math

import numpy as np
import pytest

import windowwise as ww
from windowwise.io import number_from_name, read_metadata, read_series

D, A, X0 = 30.0, 3.5, 0.7


def morse(x):
    return D * ((1 - math.exp(-A * (x - X0))) ** 2 - 1)


def morse_slope(x):
    e = math.exp(-A * (x - X0))
    return 2 * D * A * e * (1 - e)


def rmse_vs_truth(x, F, lo, hi):
    m = np.isfinite(F) & (x > lo) & (x < hi)
    t = np.array([morse(v) for v in x[m]])
    d = F[m] - t
    d -= d.mean()
    return float(np.sqrt(np.mean(d ** 2))), int(m.sum())


@pytest.fixture(scope="module")
def good_windows():
    centers = np.round(np.arange(0.5, 2.5001, 0.05), 3)
    return ww.synthetic(morse, centers, 3000.0, n=4000, stride=5, seed=3)


def test_overlap_formula():
    z = 1.0364333894937898  # -Phi^-1(0.15)
    assert ww.gauss_overlap(2 * z, 1.0) == pytest.approx(0.30, abs=1e-6)


def test_plan_rules():
    p = ww.plan(0.5, 2.5, 300, overlap=0.3, slope=60, k_min=500)
    assert p.overlap >= 0.3 - 1e-9
    assert p.shift <= 0.52 * p.spacing
    q = ww.plan(0.5, 2.5, 300, overlap=0.3, k=1000)
    assert q.sigma == pytest.approx(math.sqrt(ww.kT(300) / 1000))
    r = ww.plan(0.5, 2.5, 300, overlap=0.3, spacing=0.1)
    assert r.overlap == pytest.approx(0.3, abs=0.02)


def test_wham_recovers_exact_pmf(good_windows):
    res = ww.analyze(good_windows, 300, nbins=120, nboot=10)
    err, n = rmse_vs_truth(res.x, res.F, 0.55, 2.45)
    assert res.converged
    assert n > 90
    assert err < 0.5, err
    assert np.nanmedian(res.err) < 1.0
    assert not [f for f in res.flags if f.kind == "overlap"]


def test_gap_detected_and_repaired(good_windows):
    ws = [w for w in good_windows if not (1.3 - 1e-9 <= w.center <= 1.5 + 1e-9)]
    res = ww.analyze(ws, 300, nbins=120, nboot=0)
    assert any(f.kind == "overlap" for f in res.flags)
    assert any(1.25 < s.target < 1.55 for s in res.suggestions)


def test_soft_springs_flagged_as_drift():
    centers = np.round(np.arange(0.5, 1.0001, 0.05), 3)
    ws = ww.synthetic(morse, centers, 300.0, n=3000, seed=5)
    res = ww.analyze(ws, 300, nbins=80, nboot=0)
    assert any(f.kind == "drift" for f in res.flags)


def test_deep_pmf_log_fallback():
    def deep(x):
        return 2000 * ((1 - math.exp(-3.5 * (x - 0.7))) ** 2 - 1)
    centers = np.round(np.arange(0.6, 1.6001, 0.02), 3)
    ws = ww.synthetic(deep, centers, 60000.0, n=1500, stride=4, burn=1000, seed=9)
    res = ww.analyze(ws, 300, nbins=100, nboot=0)
    m = np.isfinite(res.F) & (res.x > 0.62) & (res.x < 1.58)
    t = np.array([deep(v) for v in res.x[m]])
    d = res.F[m] - t
    assert res.converged
    assert np.sqrt(np.mean((d - d.mean()) ** 2)) < 3.0


def test_gp_umbrella_integration(good_windows):
    g = ww.gp_pmf(good_windows[::2])              # every other window: GP must interpolate
    x = g.x
    m = (x > 0.6) & (x < 2.4)
    t = np.array([morse(v) for v in x[m]])
    d = g.A[m] - t
    assert np.sqrt(np.mean((d - d.mean()) ** 2)) < 1.0
    # mean-force accuracy away from the steep wall
    mf = np.array([morse_slope(v) for v in x[m]])
    assert np.median(np.abs(g.dA[m] - mf)) < 5.0
    assert np.all(g.A_std >= 0)


def test_gp_active_learning_targets_the_gap(good_windows):
    ws = [w for w in good_windows if not (1.3 - 1e-9 <= w.center <= 1.6 + 1e-9)]
    picks = ww.suggest_windows(ws, n=2)
    assert any(1.3 < p["target"] < 1.65 for p in picks), picks


def test_readers(tmp_path):
    f = tmp_path / "pullx_0.85.xvg"
    f.write_text('# gmx\n@ title "x"\n@TYPE xy\n0.0 0.84\n0.1 0.86\n0.2\t0.85\n')
    assert np.allclose(read_series(f), [0.84, 0.86, 0.85])
    meta = tmp_path / "meta.dat"
    meta.write_text(f"# path center k\n{f.name} 0.85 1000\n")
    rows = read_metadata(meta)
    assert rows[0][1:] == (0.85, 1000.0)
    ws = ww.load_windows(metadata=str(meta))
    assert ws[0].center == 0.85 and len(ws[0].data) == 3
    assert number_from_name("umbrella_1.35.xvg") == 1.35
    assert number_from_name("pullx-12.xvg") == 12


def test_cli_demo_runs(capsys):
    from windowwise.cli import main
    main(["plan", "0.5", "2.5", "--slope", "60"])
    out = capsys.readouterr().out
    assert "force constant k" in out and "pull-coord1-init" in out
