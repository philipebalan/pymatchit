"""Exact matching implementation."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExactMatchResult:
    """Weights and subclasses produced by exact matching."""

    weights: pd.Series
    subclass: pd.Series
    exact_columns: list[str]
    dropped_strata: int


def exact_match(
    data: pd.DataFrame,
    treatment: pd.Series,
    exact_columns: list[str],
) -> ExactMatchResult:
    """Match exactly by retaining strata with treated and control units."""
    if not exact_columns:
        raise ValueError("exact matching requires at least one exact column.")

    missing_columns = [column for column in exact_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"exact columns not found in data: {missing}.")

    weights = pd.Series(0.0, index=data.index, name="weights")
    subclass = pd.Series(pd.NA, index=data.index, dtype="Int64", name="subclass")
    dropped_strata = 0
    subclass_id = 1

    strata = data.groupby(exact_columns, sort=True, dropna=False).groups
    treatment_int = treatment.astype(int)

    for row_index in strata.values():
        stratum_treatment = treatment_int.loc[row_index]
        treated_index = stratum_treatment.index[stratum_treatment == 1]
        control_index = stratum_treatment.index[stratum_treatment == 0]

        if len(treated_index) == 0 or len(control_index) == 0:
            dropped_strata += 1
            continue

        weights.loc[treated_index] = 1.0
        weights.loc[control_index] = len(treated_index) / len(control_index)
        subclass.loc[row_index] = subclass_id
        subclass_id += 1

    return ExactMatchResult(
        weights=weights,
        subclass=subclass,
        exact_columns=exact_columns,
        dropped_strata=dropped_strata,
    )
