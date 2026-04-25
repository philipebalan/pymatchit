"""MatchIt-style diagnostic plots using matplotlib."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from pymatchit.distances import estimate_propensity_score


def love_plot(result: object, abs: bool = True, threshold: float = 0.1) -> Axes:
    """Plot standardized mean differences before and after matching."""
    summary = result.balance_summary()
    before = summary["smd_before"].copy()
    after = summary["smd_after"].copy()
    if abs:
        before = before.abs()
        after = after.abs()

    fig, ax = plt.subplots()
    y = np.arange(len(summary.index))
    ax.scatter(before, y, label="Before", marker="o")
    ax.scatter(after, y, label="After", marker="s")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    if not abs:
        ax.axvline(-threshold, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Standardized mean difference")
    ax.set_ylabel("Covariate")
    ax.legend()
    fig.tight_layout()
    return ax


def ecdf_plot(result: object, covariate: str) -> Figure:
    """Plot treated/control empirical CDFs before and after matching."""
    data = _plot_data(result, covariate)
    fig, axes = plt.subplots(1, 2, sharey=True)
    _draw_ecdf_panel(axes[0], data, "Before matching", weights=None)
    _draw_ecdf_panel(axes[1], data, "After matching", weights=data["weights"])
    fig.tight_layout()
    return fig


def qq_plot(result: object, covariate: str) -> Figure:
    """Plot treated vs control empirical quantiles before and after matching."""
    data = _plot_data(result, covariate)
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    _draw_qq_panel(axes[0], data, "Before matching", weights=None)
    _draw_qq_panel(axes[1], data, "After matching", weights=data["weights"])
    fig.tight_layout()
    return fig


def density_plot(result: object, covariate: str) -> Figure:
    """Plot weighted densities or weighted category frequencies."""
    data = _plot_data(result, covariate)
    fig, axes = plt.subplots(1, 2)
    if _is_continuous(data["value"]):
        _draw_density_panel(axes[0], data, "Before matching", weights=None)
        _draw_density_panel(axes[1], data, "After matching", weights=data["weights"])
    else:
        _draw_bar_panel(axes[0], data, "Before matching", weights=None)
        _draw_bar_panel(axes[1], data, "After matching", weights=data["weights"])
    fig.tight_layout()
    return fig


def jitter_plot(result: object) -> Axes:
    """Plot propensity scores by treatment and matched/discarded status."""
    treatment = _require_series(result, "treatment")
    propensity = _propensity_scores(result)
    weights = _weights_or_ones(result, treatment.index)
    matched = weights > 0

    fig, ax = plt.subplots()
    for group, label in [(0, "Control"), (1, "Treated")]:
        mask = treatment.astype(int) == group
        group_index = treatment.index[mask]
        offsets = _deterministic_jitter(len(group_index))
        group_matched = matched.loc[group_index]
        x = np.full(len(group_index), group, dtype=float) + offsets
        matched_index = group_index[group_matched.to_numpy()]
        discarded_index = group_index[~group_matched.to_numpy()]
        ax.scatter(
            x[group_matched.to_numpy()],
            propensity.loc[matched_index].to_numpy(),
            label=f"{label} matched",
            alpha=0.8,
        )
        ax.scatter(
            x[~group_matched.to_numpy()],
            propensity.loc[discarded_index].to_numpy(),
            label=f"{label} discarded",
            marker="x",
            alpha=0.8,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control", "Treated"])
    ax.set_ylabel("Propensity score")
    ax.legend()
    fig.tight_layout()
    return ax


def histogram_plot(result: object) -> Figure:
    """Plot propensity score histograms before and after matching."""
    treatment = _require_series(result, "treatment")
    propensity = _propensity_scores(result)
    weights = _weights_or_ones(result, treatment.index)
    data = pd.DataFrame(
        {
            "treatment": treatment.astype(int),
            "propensity": propensity,
            "weights": weights,
        }
    )

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    _draw_histogram_panel(axes[0], data, "Before matching", weights=None)
    _draw_histogram_panel(axes[1], data, "After matching", weights=data["weights"])
    fig.tight_layout()
    return fig


def _plot_data(result: object, covariate: str) -> pd.DataFrame:
    treatment = _require_series(result, "treatment")
    covariates = _require_frame(result, "covariates")
    if covariate not in covariates.columns:
        raise ValueError(f"Unknown covariate: {covariate}.")
    return pd.DataFrame(
        {
            "value": covariates[covariate],
            "treatment": treatment.astype(int),
            "weights": _weights_or_ones(result, treatment.index),
        }
    )


def _draw_ecdf_panel(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    weights: pd.Series | None,
) -> None:
    for group, label in [(0, "Control"), (1, "Treated")]:
        subset = data[data["treatment"] == group]
        group_weights = None if weights is None else weights.loc[subset.index]
        x, y = _weighted_ecdf(subset["value"], group_weights)
        ax.step(x, y, where="post", label=label)
    ax.set_title(title)
    ax.set_xlabel("Covariate value")
    ax.set_ylabel("ECDF")
    ax.legend()


def _draw_qq_panel(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    weights: pd.Series | None,
) -> None:
    probabilities = np.linspace(0.01, 0.99, 99)
    treated = data[data["treatment"] == 1]
    control = data[data["treatment"] == 0]
    treated_weights = None if weights is None else weights.loc[treated.index]
    control_weights = None if weights is None else weights.loc[control.index]
    treated_q = _weighted_quantile(treated["value"], probabilities, treated_weights)
    control_q = _weighted_quantile(control["value"], probabilities, control_weights)
    ax.scatter(control_q, treated_q, s=12)
    finite = np.concatenate(
        [
            control_q[np.isfinite(control_q)],
            treated_q[np.isfinite(treated_q)],
        ]
    )
    if finite.size:
        ax.plot(
            [finite.min(), finite.max()],
            [finite.min(), finite.max()],
            color="black",
        )
    ax.set_title(title)
    ax.set_xlabel("Control quantiles")
    ax.set_ylabel("Treated quantiles")


def _draw_density_panel(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    weights: pd.Series | None,
) -> None:
    for group, label in [(0, "Control"), (1, "Treated")]:
        subset = data[data["treatment"] == group]
        group_weights = None if weights is None else weights.loc[subset.index]
        _plot_density_or_hist(ax, subset["value"], group_weights, label)
    ax.set_title(title)
    ax.set_xlabel("Covariate value")
    ax.set_ylabel("Density")
    ax.legend()


def _draw_bar_panel(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    weights: pd.Series | None,
) -> None:
    categories = pd.Index(sorted(data["value"].dropna().unique()))
    positions = np.arange(len(categories))
    width = 0.35
    for offset, group, label in [(-width / 2, 0, "Control"), (width / 2, 1, "Treated")]:
        subset = data[data["treatment"] == group]
        group_weights = pd.Series(1.0, index=subset.index)
        if weights is not None:
            group_weights = weights.loc[subset.index]
        totals = []
        for category in categories:
            totals.append(float(group_weights[subset["value"] == category].sum()))
        denominator = sum(totals)
        heights = (
            np.array(totals) / denominator
            if denominator
            else np.zeros(len(totals))
        )
        ax.bar(positions + offset, heights, width=width, label=label)
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories.astype(str), rotation=45, ha="right")
    ax.set_ylabel("Weighted proportion")
    ax.legend()


def _draw_histogram_panel(
    ax: Axes,
    data: pd.DataFrame,
    title: str,
    weights: pd.Series | None,
) -> None:
    bins = np.linspace(0, 1, 11)
    for group, label in [(0, "Control"), (1, "Treated")]:
        subset = data[data["treatment"] == group]
        group_weights = None if weights is None else weights.loc[subset.index]
        ax.hist(
            subset["propensity"],
            bins=bins,
            weights=group_weights,
            alpha=0.45,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Count")
    ax.legend()


def _weighted_ecdf(
    values: pd.Series,
    weights: pd.Series | None,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = values.sort_values(kind="mergesort")
    if weights is None:
        ordered_weights = pd.Series(1.0, index=ordered.index)
    else:
        ordered_weights = weights.loc[ordered.index].astype(float)
    positive = ordered_weights > 0
    ordered = ordered[positive]
    ordered_weights = ordered_weights[positive]
    if ordered.empty:
        return np.array([]), np.array([])
    y = ordered_weights.cumsum() / ordered_weights.sum()
    return ordered.to_numpy(dtype=float), y.to_numpy(dtype=float)


def _weighted_quantile(
    values: pd.Series,
    probabilities: np.ndarray,
    weights: pd.Series | None,
) -> np.ndarray:
    x, cdf = _weighted_ecdf(values, weights)
    if x.size == 0:
        return np.full(len(probabilities), np.nan)
    return np.interp(probabilities, cdf, x, left=x[0], right=x[-1])


def _plot_density_or_hist(
    ax: Axes,
    values: pd.Series,
    weights: pd.Series | None,
    label: str,
) -> None:
    if weights is None:
        weights = pd.Series(1.0, index=values.index)
    positive = weights > 0
    x = values[positive].astype(float).to_numpy()
    w = weights[positive].astype(float).to_numpy()
    if x.size < 2 or np.unique(x).size < 2:
        ax.hist(x, weights=w, density=True, histtype="step", label=label)
        return
    grid = np.linspace(x.min(), x.max(), 200)
    density = gaussian_kde(x, weights=w)(grid)
    ax.plot(grid, density, label=label)


def _propensity_scores(result: object) -> pd.Series:
    propensity = getattr(result, "propensity_score", None)
    if propensity is not None:
        return propensity
    treatment = _require_series(result, "treatment")
    covariates = _require_frame(result, "covariates")
    return estimate_propensity_score(treatment, covariates)


def _weights_or_ones(result: object, index: pd.Index) -> pd.Series:
    weights = getattr(result, "weights_", None)
    if weights is None:
        return pd.Series(1.0, index=index)
    return weights.reindex(index).fillna(0.0).astype(float)


def _deterministic_jitter(size: int) -> np.ndarray:
    if size == 0:
        return np.array([])
    return np.linspace(-0.08, 0.08, size)


def _is_continuous(values: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 5


def _require_series(result: object, name: str) -> pd.Series:
    value = getattr(result, name, None)
    if value is None:
        raise NotImplementedError(f"{name} is required for this plot.")
    return value


def _require_frame(result: object, name: str) -> pd.DataFrame:
    value = getattr(result, name, None)
    if value is None:
        raise NotImplementedError(f"{name} is required for this plot.")
    return value
