"""Optimal 1:1 pair matching for binary treatment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class OptimalMatchResult:
    """Matched sets, weights, and subclasses from optimal pair matching."""

    matches: pd.DataFrame
    weights: pd.Series
    subclass: pd.Series


def optimal_pair_match(
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    caliper: float | None = None,
) -> OptimalMatchResult:
    """Perform optimal ATT 1:1 pair matching with optional forbidden edges."""
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be non-negative.")

    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]

    costs = distance_matrix.copy()
    if caliper is not None:
        costs[costs > caliper] = np.inf

    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        return _empty_result(treatment)

    penalty = float(finite_costs.max() + finite_costs.sum() + 1.0)
    size = len(treated_index) + len(control_index)
    assignment_costs = np.zeros((size, size))
    assignment_costs[: len(treated_index), : len(control_index)] = np.where(
        np.isfinite(costs),
        costs - penalty,
        penalty,
    )

    row_ind, col_ind = linear_sum_assignment(assignment_costs)
    pairs: list[tuple[int, int, float]] = []
    for row, col in zip(row_ind, col_ind, strict=True):
        if row >= len(treated_index) or col >= len(control_index):
            continue
        distance = costs[row, col]
        if np.isfinite(distance):
            pairs.append((int(row), int(col), float(distance)))

    pairs.sort(key=lambda pair: pair[0])
    return _result_from_pairs(treatment, treated_index, control_index, pairs)


def _empty_result(treatment: pd.Series) -> OptimalMatchResult:
    return OptimalMatchResult(
        matches=pd.DataFrame(
            columns=["subclass", "treated_index", "control_index", "distance"],
        ),
        weights=pd.Series(0.0, index=treatment.index, name="weights"),
        subclass=pd.Series(
            pd.NA,
            index=treatment.index,
            dtype="Int64",
            name="subclass",
        ),
    )


def _result_from_pairs(
    treatment: pd.Series,
    treated_index: pd.Index,
    control_index: pd.Index,
    pairs: list[tuple[int, int, float]],
) -> OptimalMatchResult:
    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")
    matched_sets: list[dict[str, object]] = []

    for subclass_id, (treated_position, control_position, distance) in enumerate(
        pairs,
        start=1,
    ):
        treated_label = treated_index[treated_position]
        control_label = control_index[control_position]
        weights.loc[treated_label] = 1.0
        weights.loc[control_label] = 1.0
        subclass.loc[treated_label] = subclass_id
        subclass.loc[control_label] = subclass_id
        matched_sets.append(
            {
                "subclass": subclass_id,
                "treated_index": treated_label,
                "control_index": control_label,
                "distance": distance,
            }
        )

    return OptimalMatchResult(
        matches=pd.DataFrame(
            matched_sets,
            columns=["subclass", "treated_index", "control_index", "distance"],
        ),
        weights=weights,
        subclass=subclass,
    )
