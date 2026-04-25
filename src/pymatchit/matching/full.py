"""Experimental OR-Tools prototype for full matching."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FullMatchResult:
    """Weights and subclasses from experimental full matching."""

    matches: pd.DataFrame
    weights: pd.Series
    subclass: pd.Series
    dropped_units: int


def full_match(
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    caliper: float | None = None,
) -> FullMatchResult:
    """Solve a minimal full-matching prototype with OR-Tools min-cost flow."""
    if caliper is not None and caliper < 0:
        raise ValueError("caliper must be non-negative.")

    min_cost_flow = _load_ortools_min_cost_flow()
    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]

    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")
    if len(treated_index) == 0 or len(control_index) == 0:
        return _full_result(treatment, weights, subclass, [], len(treatment))

    if len(treated_index) <= len(control_index):
        anchor_index = treated_index
        assigned_index = control_index
        costs = distance_matrix
        anchor_is_treated = True
    else:
        anchor_index = control_index
        assigned_index = treated_index
        costs = distance_matrix.T
        anchor_is_treated = False

    assignments = _solve_assignment_flow(
        min_cost_flow=min_cost_flow,
        costs=costs,
        caliper=caliper,
    )

    matched_sets = _decode_full_assignments(
        treatment=treatment_int,
        weights=weights,
        subclass=subclass,
        anchor_index=anchor_index,
        assigned_index=assigned_index,
        assignments=assignments,
        anchor_is_treated=anchor_is_treated,
    )
    dropped_units = int((weights == 0).sum())
    return _full_result(treatment, weights, subclass, matched_sets, dropped_units)


def _solve_assignment_flow(
    min_cost_flow: object,
    costs: np.ndarray,
    caliper: float | None,
) -> list[tuple[int, int, float]]:
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        return []

    feasible = np.isfinite(costs)
    if caliper is not None:
        feasible &= costs <= caliper
    if not feasible.any():
        return []

    n_assigned, n_anchor = costs.shape
    source = 0
    assigned_offset = 1
    anchor_offset = assigned_offset + n_assigned
    drop_node = anchor_offset + n_anchor
    sink = drop_node + 1
    solver = min_cost_flow.SimpleMinCostFlow()
    edge_lookup: dict[int, tuple[int, int, float]] = {}

    max_cost = float(finite_costs.max())
    penalty = _scale_cost(max_cost + finite_costs.sum() + 1.0, 0)

    for assigned_position in range(n_assigned):
        assigned_node = assigned_offset + assigned_position
        solver.add_arc_with_capacity_and_unit_cost(source, assigned_node, 1, 0)
        solver.add_arc_with_capacity_and_unit_cost(assigned_node, drop_node, 1, penalty)
        for anchor_position in range(n_anchor):
            if not feasible[assigned_position, anchor_position]:
                continue
            order = assigned_position * n_anchor + anchor_position
            solver.add_arc_with_capacity_and_unit_cost(
                assigned_node,
                anchor_offset + anchor_position,
                1,
                _scale_cost(float(costs[assigned_position, anchor_position]), order),
            )
            arc = solver.num_arcs() - 1
            edge_lookup[arc] = (
                assigned_position,
                anchor_position,
                float(costs[assigned_position, anchor_position]),
            )

    for anchor_position in range(n_anchor):
        solver.add_arc_with_capacity_and_unit_cost(
            anchor_offset + anchor_position,
            sink,
            n_assigned,
            0,
        )
    solver.add_arc_with_capacity_and_unit_cost(drop_node, sink, n_assigned, 0)
    solver.set_node_supply(source, n_assigned)
    solver.set_node_supply(sink, -n_assigned)

    status = solver.solve()
    if status != solver.OPTIMAL:
        raise RuntimeError("OR-Tools min-cost flow did not find an optimal solution.")

    assignments: list[tuple[int, int, float]] = []
    for arc in range(solver.num_arcs()):
        if solver.flow(arc) != 1 or arc not in edge_lookup:
            continue
        assignments.append(edge_lookup[arc])
    assignments.sort(key=lambda item: (item[1], item[0]))
    return assignments


def _decode_full_assignments(
    treatment: pd.Series,
    weights: pd.Series,
    subclass: pd.Series,
    anchor_index: pd.Index,
    assigned_index: pd.Index,
    assignments: list[tuple[int, int, float]],
    anchor_is_treated: bool,
) -> list[dict[str, object]]:
    by_anchor: dict[int, list[tuple[int, float]]] = {}
    for assigned_position, anchor_position, distance in assignments:
        by_anchor.setdefault(anchor_position, []).append((assigned_position, distance))

    matched_sets: list[dict[str, object]] = []
    for subclass_id, anchor_position in enumerate(sorted(by_anchor), start=1):
        anchor_label = anchor_index[anchor_position]
        assigned_labels = [
            assigned_index[position] for position, _ in by_anchor[anchor_position]
        ]
        unit_labels = [anchor_label, *assigned_labels]
        n_treated = int((treatment.loc[unit_labels] == 1).sum())
        n_control = int((treatment.loc[unit_labels] == 0).sum())
        if n_treated == 0 or n_control == 0:
            continue

        unit_treatment = treatment.loc[unit_labels]
        treated_units = unit_treatment[unit_treatment == 1].index
        control_units = unit_treatment[unit_treatment == 0].index
        weights.loc[treated_units] = 1.0
        control_weight = n_treated / n_control
        weights.loc[control_units] = control_weight
        subclass.loc[unit_labels] = subclass_id
        matched_sets.append(
            {
                "subclass": subclass_id,
                "n_treated": n_treated,
                "n_control": n_control,
                "anchor_treatment": 1 if anchor_is_treated else 0,
                "total_distance": sum(
                    distance for _, distance in by_anchor[anchor_position]
                ),
            }
        )
    return matched_sets


def _full_result(
    treatment: pd.Series,
    weights: pd.Series,
    subclass: pd.Series,
    matched_sets: list[dict[str, object]],
    dropped_units: int,
) -> FullMatchResult:
    return FullMatchResult(
        matches=pd.DataFrame(
            matched_sets,
            columns=[
                "subclass",
                "n_treated",
                "n_control",
                "anchor_treatment",
                "total_distance",
            ],
        ),
        weights=weights.reindex(treatment.index).fillna(0.0),
        subclass=subclass,
        dropped_units=dropped_units,
    )


def _scale_cost(distance: float, order: int) -> int:
    return int(round(distance * 1_000_000)) * 1_000 + order


def _load_ortools_min_cost_flow() -> object:
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError as exc:
        raise ImportError(
            "method='full' requires OR-Tools. Install with "
            "`pip install -e .[full]` or `pip install ortools`."
        ) from exc
    return min_cost_flow
