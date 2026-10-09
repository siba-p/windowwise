"""WindowWise: plan, check and repair umbrella-sampling simulations."""
__version__ = "0.1.0"

from .analysis import Analysis, analyze, synthetic  # noqa: E402
from .core import Bins, Plan, Window, bootstrap_errors, gauss_overlap, kT, plan, statistical_inefficiency, wham  # noqa: E402
from .gp import gp_pmf, suggest_windows  # noqa: E402
from .io import load_windows, mdp_snippet, read_metadata, read_series  # noqa: E402

__all__ = ["Analysis", "Bins", "Plan", "Window", "analyze", "bootstrap_errors", "gauss_overlap", "gp_pmf", "kT",
           "load_windows", "mdp_snippet", "plan", "read_metadata", "read_series", "statistical_inefficiency",
           "suggest_windows", "synthetic", "wham"]
