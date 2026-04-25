"""Nearest-neighbor matching for binary treatment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NearestMatchResult:
    """Matched sets, weights, and subclasses from nearest-neighbor matching."""

    matches: pd.DataFrame
    weights: pd.Series
    subclass: pd.Series


def nearest_neighbor_match(
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    replace: bool = False,
    ratio: int = 1,
    caliper: float | None = None,
) -> NearestMatchResult:
    """Perform ATT nearest-neighbor matching for binary treatment."""
    if ratio != 1:
        raise NotImplementedError("Only ratio=1 is currently supported.")
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be non-negative.")

    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]

    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")

    available_controls = np.ones(len(control_index), dtype=bool)
    matched_sets: list[dict[str, object]] = []
    subclass_id = 1

    for treated_position, treated_label in enumerate(treated_index):
        distances = distance_matrix[treated_position].copy()
        if caliper is not None:
            distances[distances > caliper] = np.inf
        if not replace:
            distances[~available_controls] = np.inf

        control_position = int(np.argmin(distances))
        match_distance = float(distances[control_position])
        if not np.isfinite(match_distance):
            continue

        control_label = control_index[control_position]
        matched_sets.append(
            {
                "subclass": subclass_id,
                "treated_index": treated_label,
                "control_index": control_label,
                "distance": match_distance,
            }
        )

        weights.loc[treated_label] = 1.0
        weights.loc[control_label] += 1.0
        subclass.loc[treated_label] = subclass_id
        if pd.isna(subclass.loc[control_label]):
            subclass.loc[control_label] = subclass_id

        if not replace:
            available_controls[control_position] = False
        subclass_id += 1

    return NearestMatchResult(
        matches=pd.DataFrame(
            matched_sets,
            columns=["subclass", "treated_index", "control_index", "distance"],
        ),
        weights=weights,
        subclass=subclass,
    )
