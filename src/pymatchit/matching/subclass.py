"""Propensity score subclassification."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SubclassMatchResult:
    """Weights and subclasses from propensity score subclassification."""

    strata: pd.DataFrame
    weights: pd.Series
    subclass: pd.Series
    dropped_subclasses: int


def subclass_match(
    treatment: pd.Series,
    propensity_score: pd.Series,
    n_subclasses: int = 6,
    estimand: str = "ATT",
) -> SubclassMatchResult:
    """Create quantile subclasses and compute ATT or ATE-style weights."""
    if n_subclasses < 1:
        raise ValueError("n_subclasses must be at least 1.")
    if estimand not in {"ATT", "ATE"}:
        raise NotImplementedError("Subclassification supports ATT and ATE.")

    treatment_int = treatment.astype(int)
    bins = _quantile_bins(propensity_score, n_subclasses)
    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")

    strata_rows: list[dict[str, float | int]] = []
    dropped_subclasses = 0
    subclass_id = 1

    for bin_id in sorted(pd.unique(bins.dropna())):
        row_index = bins.index[bins == bin_id]
        stratum_treatment = treatment_int.loc[row_index]
        treated_index = stratum_treatment.index[stratum_treatment == 1]
        control_index = stratum_treatment.index[stratum_treatment == 0]
        n_treated = len(treated_index)
        n_control = len(control_index)

        if n_treated == 0 or n_control == 0:
            dropped_subclasses += 1
            continue

        if estimand == "ATT":
            weights.loc[treated_index] = 1.0
            weights.loc[control_index] = n_treated / n_control
        else:
            stratum_size = n_treated + n_control
            weights.loc[treated_index] = stratum_size / n_treated
            weights.loc[control_index] = stratum_size / n_control

        subclass.loc[row_index] = subclass_id
        strata_rows.append(
            {
                "subclass": subclass_id,
                "n_treated": n_treated,
                "n_control": n_control,
                "propensity_min": float(propensity_score.loc[row_index].min()),
                "propensity_max": float(propensity_score.loc[row_index].max()),
            }
        )
        subclass_id += 1

    return SubclassMatchResult(
        strata=pd.DataFrame(
            strata_rows,
            columns=[
                "subclass",
                "n_treated",
                "n_control",
                "propensity_min",
                "propensity_max",
            ],
        ),
        weights=weights,
        subclass=subclass,
        dropped_subclasses=dropped_subclasses,
    )


def _quantile_bins(propensity_score: pd.Series, n_subclasses: int) -> pd.Series:
    ranked_scores = propensity_score.rank(method="first")
    bins = pd.qcut(
        ranked_scores,
        q=n_subclasses,
        labels=False,
        duplicates="drop",
    )
    return pd.Series(bins, index=propensity_score.index, name="subclass_bin")
