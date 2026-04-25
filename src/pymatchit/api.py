"""Top-level user-facing API for pymatchit."""

import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import patsy

from pymatchit.diagnostics import balance_summary, effective_sample_size
from pymatchit.distances import compute_distance_matrix, estimate_propensity_score
from pymatchit.matching.exact import exact_match
from pymatchit.matching.full import full_match
from pymatchit.matching.nearest import nearest_neighbor_match
from pymatchit.matching.optimal import optimal_pair_match
from pymatchit.matching.subclass import subclass_match


SUPPORTED_METHODS = {"nearest", "exact", "full", "optimal", "subclass"}
SUPPORTED_ESTIMANDS = {"ATT", "ATE"}


@dataclass(frozen=True)
class ParsedFormula:
    """Design matrices and treatment assignment extracted from a formula."""

    treatment_name: str
    treatment: pd.Series
    covariates: pd.DataFrame
    covariate_names: list[str]


@dataclass(frozen=True)
class MatchResult:
    """Container for future matching results."""

    method: str
    estimand: str = "ATT"
    formula: str | None = None
    treatment: pd.Series | None = None
    covariates: pd.DataFrame | None = None
    propensity_score: pd.Series | None = None
    distance_matrix: np.ndarray | None = None
    matches: Any | None = None
    diagnostics: Any | None = None
    weights: Any | None = None
    data: pd.DataFrame | None = None
    weights_: pd.Series | None = None
    subclass_: pd.Series | None = None

    def matched_data(self) -> pd.DataFrame:
        """Return rows retained by matching with weights and subclass columns."""
        if self.data is None or self.weights_ is None or self.subclass_ is None:
            raise NotImplementedError("matched_data is only available after matching.")

        matched = self.data.loc[self.weights_ > 0].copy()
        matched["weights"] = self.weights_.loc[matched.index]
        matched["subclass"] = self.subclass_.loc[matched.index]
        return matched

    def balance_summary(self) -> pd.DataFrame:
        """Return covariate balance before and after matching."""
        if self.covariates is None or self.treatment is None:
            raise NotImplementedError("balance_summary requires parsed covariates.")
        return balance_summary(self.covariates, self.treatment, self.weights_)

    def effective_sample_size(self) -> dict[str, float]:
        """Return effective sample sizes after matching."""
        if self.treatment is None:
            raise NotImplementedError("effective_sample_size requires treatment data.")
        return effective_sample_size(self.treatment, self.weights_)

    def match_summary(self) -> dict[str, Any]:
        """Return a compact summary of matched and unmatched units."""
        if self.weights_ is None:
            raise NotImplementedError("match_summary is only available after matching.")

        matched_units = int((self.weights_ > 0).sum())
        total_units = int(len(self.weights_))
        subclasses = 0
        if self.subclass_ is not None:
            subclasses = int(self.subclass_.dropna().nunique())
        summary = {
            "method": self.method,
            "estimand": self.estimand,
            "n_total": total_units,
            "n_matched": matched_units,
            "n_unmatched": total_units - matched_units,
            "n_subclasses": subclasses,
        }
        if self.diagnostics is not None and "dropped_strata" in self.diagnostics:
            summary["dropped_strata"] = self.diagnostics["dropped_strata"]
        if self.diagnostics is not None and "dropped_subclasses" in self.diagnostics:
            summary["dropped_subclasses"] = self.diagnostics["dropped_subclasses"]
        if self.diagnostics is not None and "dropped_units" in self.diagnostics:
            summary["dropped_units"] = self.diagnostics["dropped_units"]
        if self.treatment is not None:
            ess = self.effective_sample_size()
            summary["ess_treated"] = ess["treated"]
            summary["ess_control"] = ess["control"]
        return summary


def parse_formula(formula: str, data: pd.DataFrame) -> ParsedFormula:
    """Parse a treatment formula into treatment and covariate design matrices."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    y, x = patsy.dmatrices(formula, data=data, return_type="dataframe")
    if y.shape[1] != 1:
        raise ValueError("formula must have exactly one treatment variable.")

    treatment_name = y.columns[0]
    treatment = y.iloc[:, 0]
    unique_values = set(pd.unique(treatment.dropna()))
    if unique_values - {0, 1, 0.0, 1.0, False, True}:
        raise ValueError("treatment must be binary and coded as 0/1.")
    if len(unique_values) != 2:
        raise ValueError("treatment must contain both treated and control units.")

    covariates = x.drop(columns=["Intercept"], errors="ignore")
    if covariates.shape[1] == 0:
        raise ValueError("formula must include at least one covariate.")

    covariate_names = _formula_covariate_names(formula, data, treatment_name)

    return ParsedFormula(
        treatment_name=treatment_name,
        treatment=treatment.astype(int),
        covariates=covariates,
        covariate_names=covariate_names,
    )


def matchit(
    formula: str,
    data: pd.DataFrame,
    method: str,
    distance: str = "logit",
    estimand: str = "ATT",
    exact: bool | str | list[str] | tuple[str, ...] | None = None,
    replace: bool = False,
    ratio: int = 1,
    caliper: float | None = None,
    n_subclasses: int = 6,
    **kwargs: Any,
) -> MatchResult:
    """Prepare design-stage inputs for a future matching method.

    Matching methods are intentionally still stubs. This function parses the
    formula, estimates optional propensity scores, and computes the requested
    treated-by-control distance matrix.
    """
    del kwargs

    if method not in SUPPORTED_METHODS:
        supported = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"method must be one of: {supported}.")
    if estimand not in SUPPORTED_ESTIMANDS:
        supported_estimands = ", ".join(sorted(SUPPORTED_ESTIMANDS))
        raise ValueError(f"estimand must be one of: {supported_estimands}.")
    if method != "subclass" and estimand != "ATT":
        raise NotImplementedError("Only subclassification currently supports ATE.")

    parsed = parse_formula(formula, data)
    parsed_data = data.loc[parsed.treatment.index].copy()

    if method == "subclass":
        propensity_score = estimate_propensity_score(
            parsed.treatment,
            parsed.covariates,
        )
        subclass_result = subclass_match(
            treatment=parsed.treatment,
            propensity_score=propensity_score,
            n_subclasses=n_subclasses,
            estimand=estimand,
        )
        return MatchResult(
            method=method,
            estimand=estimand,
            formula=formula,
            treatment=parsed.treatment,
            covariates=parsed.covariates,
            propensity_score=propensity_score,
            matches=subclass_result.strata,
            data=parsed_data,
            weights=subclass_result.weights,
            weights_=subclass_result.weights,
            subclass_=subclass_result.subclass,
            diagnostics={
                "n_subclasses_requested": n_subclasses,
                "dropped_subclasses": subclass_result.dropped_subclasses,
            },
        )

    if method == "exact":
        exact_columns = _resolve_exact_columns(exact, parsed)
        exact_result = exact_match(parsed_data, parsed.treatment, exact_columns)
        return MatchResult(
            method=method,
            estimand=estimand,
            formula=formula,
            treatment=parsed.treatment,
            covariates=parsed.covariates,
            data=parsed_data,
            weights=exact_result.weights,
            weights_=exact_result.weights,
            subclass_=exact_result.subclass,
            diagnostics={
                "exact_columns": exact_result.exact_columns,
                "dropped_strata": exact_result.dropped_strata,
            },
        )

    propensity_score = None
    if distance in {"propensity", "logit"}:
        propensity_score = estimate_propensity_score(
            parsed.treatment,
            parsed.covariates,
        )

    distance_matrix = compute_distance_matrix(
        treatment=parsed.treatment,
        covariates=parsed.covariates,
        distance=distance,
        propensity_score=propensity_score,
    )

    if method == "nearest":
        if exact is not None and exact is not False:
            exact_columns = _resolve_exact_columns(exact, parsed)
            distance_matrix = _apply_exact_blocking(
                data=parsed_data,
                treatment=parsed.treatment,
                distance_matrix=distance_matrix,
                exact_columns=exact_columns,
            )

        nearest_result = nearest_neighbor_match(
            treatment=parsed.treatment,
            distance_matrix=distance_matrix,
            replace=replace,
            ratio=ratio,
            caliper=caliper,
        )
        return MatchResult(
            method=method,
            estimand=estimand,
            formula=formula,
            treatment=parsed.treatment,
            covariates=parsed.covariates,
            propensity_score=propensity_score,
            distance_matrix=distance_matrix,
            matches=nearest_result.matches,
            data=parsed_data,
            weights=nearest_result.weights,
            weights_=nearest_result.weights,
            subclass_=nearest_result.subclass,
            diagnostics={
                "replace": replace,
                "ratio": ratio,
                "caliper": caliper,
            },
        )

    if method == "optimal":
        if replace:
            raise NotImplementedError("Optimal matching is 1:1 without replacement.")
        if ratio != 1:
            raise NotImplementedError("Only ratio=1 is currently supported.")

        exact_columns = None
        if exact is not None and exact is not False:
            exact_columns = _resolve_exact_columns(exact, parsed)
            distance_matrix = _apply_exact_blocking(
                data=parsed_data,
                treatment=parsed.treatment,
                distance_matrix=distance_matrix,
                exact_columns=exact_columns,
            )

        optimal_result = _optimal_match_with_optional_blocks(
            data=parsed_data,
            treatment=parsed.treatment,
            distance_matrix=distance_matrix,
            caliper=caliper,
            exact_columns=exact_columns,
        )
        return MatchResult(
            method=method,
            estimand=estimand,
            formula=formula,
            treatment=parsed.treatment,
            covariates=parsed.covariates,
            propensity_score=propensity_score,
            distance_matrix=distance_matrix,
            matches=optimal_result.matches,
            data=parsed_data,
            weights=optimal_result.weights,
            weights_=optimal_result.weights,
            subclass_=optimal_result.subclass,
            diagnostics={
                "ratio": ratio,
                "caliper": caliper,
                "exact_columns": exact_columns,
            },
        )

    if method == "full":
        if replace:
            raise NotImplementedError("Full matching does not support replace=True.")
        warnings.warn(
            "method='full' is an experimental OR-Tools prototype.",
            UserWarning,
            stacklevel=2,
        )

        exact_columns = None
        if exact is not None and exact is not False:
            exact_columns = _resolve_exact_columns(exact, parsed)

        full_result = _full_match_with_optional_blocks(
            data=parsed_data,
            treatment=parsed.treatment,
            distance_matrix=distance_matrix,
            caliper=caliper,
            exact_columns=exact_columns,
        )
        return MatchResult(
            method=method,
            estimand=estimand,
            formula=formula,
            treatment=parsed.treatment,
            covariates=parsed.covariates,
            propensity_score=propensity_score,
            distance_matrix=distance_matrix,
            matches=full_result.matches,
            data=parsed_data,
            weights=full_result.weights,
            weights_=full_result.weights,
            subclass_=full_result.subclass,
            diagnostics={
                "experimental": True,
                "solver": "ortools",
                "caliper": caliper,
                "exact_columns": exact_columns,
                "dropped_units": full_result.dropped_units,
            },
        )

    return MatchResult(
        method=method,
        estimand=estimand,
        formula=formula,
        treatment=parsed.treatment,
        covariates=parsed.covariates,
        propensity_score=propensity_score,
        distance_matrix=distance_matrix,
    )


def match(*args: Any, **kwargs: Any) -> MatchResult:
    """Backward-compatible alias for :func:`matchit`."""
    return matchit(*args, **kwargs)


def _resolve_exact_columns(
    exact: bool | str | list[str] | tuple[str, ...] | None,
    parsed: ParsedFormula,
) -> list[str]:
    if exact is None:
        return parsed.covariate_names
    if exact is True:
        raise ValueError(
            "exact=True is ambiguous with formula transformations; "
            "pass exact=[...] explicitly."
        )
    if exact is False:
        raise ValueError("method='exact' requires exact=True or a list of columns.")
    if isinstance(exact, str):
        return [exact]
    return list(exact)


def _formula_covariate_names(
    formula: str,
    data: pd.DataFrame,
    treatment_name: str,
) -> list[str]:
    description = patsy.ModelDesc.from_formula(formula)
    covariate_names: list[str] = []
    for term in description.rhs_termlist:
        for factor in term.factors:
            name = factor.name()
            if name in data.columns and name != treatment_name:
                covariate_names.append(name)
    return list(dict.fromkeys(covariate_names))


def _apply_exact_blocking(
    data: pd.DataFrame,
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    exact_columns: list[str],
) -> np.ndarray:
    missing_columns = [column for column in exact_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"exact columns not found in data: {missing}.")

    blocked = distance_matrix.copy()
    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]

    for treated_position, treated_label in enumerate(treated_index):
        treated_values = data.loc[treated_label, exact_columns]
        for control_position, control_label in enumerate(control_index):
            control_values = data.loc[control_label, exact_columns]
            if not treated_values.equals(control_values):
                blocked[treated_position, control_position] = np.inf

    return blocked


def _optimal_match_with_optional_blocks(
    data: pd.DataFrame,
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    caliper: float | None,
    exact_columns: list[str] | None,
) -> Any:
    if exact_columns is None:
        return optimal_pair_match(treatment, distance_matrix, caliper=caliper)

    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]
    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")
    matched_sets: list[dict[str, object]] = []
    subclass_id = 1

    grouped = data.groupby(exact_columns, sort=True, dropna=False).groups
    for row_index in grouped.values():
        block_treatment = treatment.loc[row_index]
        block_treated = block_treatment.index[block_treatment == 1]
        block_controls = block_treatment.index[block_treatment == 0]
        if len(block_treated) == 0 or len(block_controls) == 0:
            continue

        treated_positions = [treated_index.get_loc(index) for index in block_treated]
        control_positions = [control_index.get_loc(index) for index in block_controls]
        block_distances = distance_matrix[np.ix_(treated_positions, control_positions)]
        block_result = optimal_pair_match(
            block_treatment,
            block_distances,
            caliper=caliper,
        )

        for _, match_row in block_result.matches.iterrows():
            treated_label = match_row["treated_index"]
            control_label = match_row["control_index"]
            weights.loc[treated_label] = 1.0
            weights.loc[control_label] = 1.0
            subclass.loc[treated_label] = subclass_id
            subclass.loc[control_label] = subclass_id
            matched_sets.append(
                {
                    "subclass": subclass_id,
                    "treated_index": treated_label,
                    "control_index": control_label,
                    "distance": match_row["distance"],
                }
            )
            subclass_id += 1

    return SimpleNamespace(
        matches=pd.DataFrame(
            matched_sets,
            columns=["subclass", "treated_index", "control_index", "distance"],
        ),
        weights=weights,
        subclass=subclass,
    )


def _full_match_with_optional_blocks(
    data: pd.DataFrame,
    treatment: pd.Series,
    distance_matrix: np.ndarray,
    caliper: float | None,
    exact_columns: list[str] | None,
) -> Any:
    if exact_columns is None:
        return full_match(treatment, distance_matrix, caliper=caliper)

    treatment_int = treatment.astype(int)
    treated_index = treatment_int.index[treatment_int == 1]
    control_index = treatment_int.index[treatment_int == 0]
    weights = pd.Series(0.0, index=treatment.index, name="weights")
    subclass = pd.Series(pd.NA, index=treatment.index, dtype="Int64", name="subclass")
    matched_sets: list[dict[str, object]] = []
    subclass_id = 1

    grouped = data.groupby(exact_columns, sort=True, dropna=False).groups
    for row_index in grouped.values():
        block_treatment = treatment.loc[row_index]
        block_treated = block_treatment.index[block_treatment == 1]
        block_controls = block_treatment.index[block_treatment == 0]
        if len(block_treated) == 0 or len(block_controls) == 0:
            continue

        treated_positions = [treated_index.get_loc(index) for index in block_treated]
        control_positions = [control_index.get_loc(index) for index in block_controls]
        block_distances = distance_matrix[np.ix_(treated_positions, control_positions)]
        block_result = full_match(
            block_treatment,
            block_distances,
            caliper=caliper,
        )

        for _, match_row in block_result.matches.iterrows():
            unit_index = block_result.subclass.index[
                block_result.subclass == match_row["subclass"]
            ]
            weights.loc[unit_index] = block_result.weights.loc[unit_index]
            subclass.loc[unit_index] = subclass_id
            matched_sets.append(
                {
                    "subclass": subclass_id,
                    "n_treated": match_row["n_treated"],
                    "n_control": match_row["n_control"],
                    "anchor_treatment": match_row["anchor_treatment"],
                    "total_distance": match_row["total_distance"],
                }
            )
            subclass_id += 1

    return SimpleNamespace(
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
        weights=weights,
        subclass=subclass,
        dropped_units=int((weights == 0).sum()),
    )
