"""Readers for GROMACS pullx.xvg files and Grossfield-style WHAM metadata, and .mdp writers."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from .core import Window

_NUM = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?(?:[eE]-?\d+)?|\d+(?:\.\d+)?")


def read_series(path, column: int = 1) -> np.ndarray:
    """Read one column from an .xvg or whitespace/comma separated file, skipping '#', '@' and '&' lines."""
    vals = []
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t or t[0] in "#@&":
                continue
            parts = re.split(r"[\s,]+", t)
            c = column if len(parts) > column else len(parts) - 1
            try:
                vals.append(float(parts[c]))
            except ValueError:
                continue
    return np.asarray(vals)


def read_metadata(path):
    """Grossfield WHAM metadata: lines of 'path center k [corr_time]'. Returns list of (path, center, k)."""
    rows = []
    base = Path(path).parent
    with open(path) as fh:
        for line in fh:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            p = t.split()
            if len(p) < 3:
                raise ValueError(f"bad metadata line: {line!r}")
            fp = Path(p[0])
            if not fp.is_absolute() and not fp.exists():
                fp = base / fp
            if not fp.exists():
                fp = base / Path(p[0]).name
            rows.append((str(fp), float(p[1]), float(p[2])))
    return rows


def number_from_name(name: str) -> float:
    m = _NUM.findall(os.path.splitext(os.path.basename(name))[0])
    return float(m[-1]) if m else float("nan")


def load_windows(files=None, metadata=None, k=None, centers=None, column: int = 1) -> list[Window]:
    """Build windows from a metadata file, or from files plus k and centres (explicit or from file names)."""
    if metadata:
        return [Window(read_series(p, column), c, kk, os.path.basename(p)) for p, c, kk in read_metadata(metadata)]
    if not files:
        raise ValueError("give metadata or files")
    if k is None:
        raise ValueError("give the force constant k (or a metadata file)")
    ws = []
    for i, f in enumerate(files):
        c = centers[i] if centers is not None else number_from_name(f)
        if not np.isfinite(c):
            raise ValueError(f"cannot read a centre from file name {f!r}; pass centres or a metadata file")
        ws.append(Window(read_series(f, column), float(c), float(k), os.path.basename(f)))
    return sorted(ws, key=lambda w: w.center)


def mdp_snippet(centers, ks, geometry: str = "distance", header: str = "umbrella windows") -> str:
    lines = [
        f"; WindowWise: {header} (GROMACS pull code, U = 0.5 k (x - x0)^2)",
        "pull                 = yes",
        "pull-ncoords         = 1",
        "pull-ngroups         = 2",
        "pull-group1-name     = GROUP_A",
        "pull-group2-name     = GROUP_B",
        "pull-coord1-type     = umbrella",
        f"pull-coord1-geometry = {geometry}",
        "pull-coord1-groups   = 1 2",
        "pull-coord1-rate     = 0.0",
        "",
        "; per-window settings (one .mdp per window):",
    ]
    for i, (c, kk) in enumerate(zip(centers, ks)):
        lines.append(f"; window {i:02d}:  pull-coord1-init = {c:.4f}   pull-coord1-k = {round(kk)}")
    return "\n".join(lines) + "\n"
