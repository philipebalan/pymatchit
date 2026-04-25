"""Public package interface for pymatchit."""

from pymatchit.api import MatchResult, ParsedFormula, match, matchit, parse_formula
from pymatchit.diagnostics import (
    balance_summary,
    effective_sample_size,
    standardized_mean_difference,
    variance_ratio,
)
from pymatchit.plots import (
    density_plot,
    ecdf_plot,
    histogram_plot,
    jitter_plot,
    love_plot,
    qq_plot,
)

__all__ = [
    "MatchResult",
    "ParsedFormula",
    "balance_summary",
    "effective_sample_size",
    "density_plot",
    "ecdf_plot",
    "histogram_plot",
    "jitter_plot",
    "love_plot",
    "match",
    "matchit",
    "parse_formula",
    "qq_plot",
    "standardized_mean_difference",
    "variance_ratio",
]
