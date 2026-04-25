"""Balance diagnostics for matching results."""

import numpy as np
import pandas as pd


def balance_summary(
    covariates: pd.DataFrame,
    treatment: pd.Series,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Return before/after balance statistics for each covariate."""
    treatment_int = treatment.astype(int)
    before_weights = pd.Series(1.0, index=treatment.index)
    after_weights = _aligned_weights(treatment, weights)

    rows: list[dict[str, float | str]] = []
    for column in covariates.columns:
        values = covariates[column].astype(float)
        denominator = _weighted_std(
            values[treatment_int == 1],
            before_weights[treatment_int == 1],
        )

        before = _balance_for_covariate(
            values,
            treatment_int,
            before_weights,
            denominator,
        )
        after = _balance_for_covariate(
            values,
            treatment_int,
            after_weights,
            denominator,
        )
        rows.append(
            {
                "variable": column,
                "mean_treated_before": before["mean_treated"],
                "mean_control_before": before["mean_control"],
                "smd_before": before["smd"],
                "variance_ratio_before": before["variance_ratio"],
                "mean_treated_after": after["mean_treated"],
                "mean_control_after": after["mean_control"],
                "smd_after": after["smd"],
                "variance_ratio_after": after["variance_ratio"],
            }
        )

    return pd.DataFrame(rows).set_index("variable")


def standardized_mean_difference(
    values: pd.Series,
    treatment: pd.Series,
    weights: pd.Series | None = None,
) -> float:
    """Return the ATT standardized mean difference for one covariate."""
    treatment_int = treatment.astype(int)
    before_weights = pd.Series(1.0, index=treatment.index)
    analysis_weights = _aligned_weights(treatment, weights)
    denominator = _weighted_std(
        values[treatment_int == 1],
        before_weights[treatment_int == 1],
    )
    return _balance_for_covariate(
        values.astype(float),
        treatment_int,
        analysis_weights,
        denominator,
    )["smd"]


def variance_ratio(
    values: pd.Series,
    treatment: pd.Series,
    weights: pd.Series | None = None,
) -> float:
    """Return treated/control variance ratio for one covariate."""
    treatment_int = treatment.astype(int)
    analysis_weights = _aligned_weights(treatment, weights)
    treated_mask = treatment_int == 1
    control_mask = treatment_int == 0
    treated_var = _weighted_var(values[treated_mask], analysis_weights[treated_mask])
    control_var = _weighted_var(values[control_mask], analysis_weights[control_mask])
    return _safe_ratio(treated_var, control_var)


def effective_sample_size(
    treatment: pd.Series,
    weights: pd.Series | None = None,
) -> dict[str, float]:
    """Return effective sample sizes by treatment group."""
    treatment_int = treatment.astype(int)
    analysis_weights = _aligned_weights(treatment, weights)
    treated_weights = analysis_weights[treatment_int == 1]
    control_weights = analysis_weights[treatment_int == 0]
    return {
        "treated": _ess(treated_weights),
        "control": _ess(control_weights),
        "total": _ess(analysis_weights),
    }


def _balance_for_covariate(
    values: pd.Series,
    treatment: pd.Series,
    weights: pd.Series,
    denominator: float,
) -> dict[str, float]:
    treated_mask = treatment == 1
    control_mask = treatment == 0
    treated_weights = weights[treated_mask]
    control_weights = weights[control_mask]
    treated_values = values[treated_mask]
    control_values = values[control_mask]

    mean_treated = _weighted_mean(treated_values, treated_weights)
    mean_control = _weighted_mean(control_values, control_weights)
    var_treated = _weighted_var(treated_values, treated_weights)
    var_control = _weighted_var(control_values, control_weights)
    return {
        "mean_treated": mean_treated,
        "mean_control": mean_control,
        "smd": _safe_ratio(mean_treated - mean_control, denominator),
        "variance_ratio": _safe_ratio(var_treated, var_control),
    }


def _aligned_weights(treatment: pd.Series, weights: pd.Series | None) -> pd.Series:
    if weights is None:
        return pd.Series(1.0, index=treatment.index)
    return weights.reindex(treatment.index).fillna(0.0).astype(float)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    positive = weights > 0
    if not positive.any():
        return np.nan
    return float(np.average(values[positive], weights=weights[positive]))


def _weighted_var(values: pd.Series, weights: pd.Series) -> float:
    positive = weights > 0
    if not positive.any():
        return np.nan
    selected_values = values[positive].to_numpy(dtype=float)
    selected_weights = weights[positive].to_numpy(dtype=float)
    mean = np.average(selected_values, weights=selected_weights)
    return float(np.average((selected_values - mean) ** 2, weights=selected_weights))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    return float(np.sqrt(_weighted_var(values, weights)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator):
        return np.nan
    if denominator == 0:
        if numerator == 0:
            return 0.0
        return np.inf
    return float(numerator / denominator)


def _ess(weights: pd.Series) -> float:
    positive = weights[weights > 0]
    if positive.empty:
        return 0.0
    return float((positive.sum() ** 2) / (positive**2).sum())
