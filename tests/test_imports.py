# ruff: noqa: E402
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import pymatchit
from pymatchit.matching.nearest import nearest_neighbor_match
from pymatchit.matching.optimal import optimal_pair_match
from pymatchit.matching.subclass import subclass_match


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [1, 0, 1, 0, 0],
            "age": [31, 42, 29, 55, 48],
            "income": [70_000, 62_000, 80_000, 59_000, 64_000],
            "x1": [0.2, -1.1, 1.4, 0.0, -0.4],
        }
    )


def make_exact_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [1, 0, 1, 0, 1, 0],
            "sex": ["F", "F", "M", "M", "F", "M"],
            "region": ["N", "N", "S", "S", "S", "W"],
            "age": [31, 42, 29, 55, 48, 36],
        },
        index=["a", "b", "c", "d", "e", "f"],
    )


def make_nearest_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "x": [0.0, 0.2, 0.1, 5.0],
        },
        index=["t1", "t2", "c1", "c2"],
    )


def make_plot_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [1, 1, 1, 1, 0, 0, 0, 0],
            "x": [0.0, 0.4, 0.8, 1.2, 0.1, 0.5, 1.5, 2.0],
            "z": [2.0, 1.8, 3.0, 2.7, 2.1, 2.0, 3.4, 3.8],
            "sex": ["F", "M", "F", "M", "F", "M", "F", "M"],
        }
    )


def test_package_exports_match_api() -> None:
    assert pymatchit.MatchResult(method="nearest").method == "nearest"


def test_formula_parsing_extracts_covariates() -> None:
    parsed = pymatchit.parse_formula("treat ~ age + income + x1", make_data())

    assert parsed.treatment_name == "treat"
    assert list(parsed.covariates.columns) == ["age", "income", "x1"]
    assert parsed.covariates.shape == (5, 3)


def test_formula_parsing_extracts_binary_treatment() -> None:
    parsed = pymatchit.parse_formula("treat ~ age + income + x1", make_data())

    assert parsed.treatment.tolist() == [1, 0, 1, 0, 0]
    assert pd.api.types.is_integer_dtype(parsed.treatment)


def test_matchit_propensity_distance_matrix_shape() -> None:
    result = pymatchit.matchit(
        "treat ~ age + income + x1",
        make_data(),
        method="nearest",
        distance="propensity",
    )

    assert result.matches is not None
    assert result.distance_matrix is not None
    assert result.distance_matrix.shape == (2, 3)


def test_matchit_logit_distance_matrix_shape() -> None:
    result = pymatchit.matchit(
        "treat ~ age + income + x1",
        make_data(),
        method="nearest",
        distance="logit",
    )

    assert result.distance_matrix is not None
    assert result.distance_matrix.shape == (2, 3)


def test_matchit_mahalanobis_distance_matrix_shape() -> None:
    result = pymatchit.matchit(
        "treat ~ age + income + x1",
        make_data(),
        method="nearest",
        distance="mahalanobis",
    )

    assert result.propensity_score is None
    assert result.distance_matrix is not None
    assert result.distance_matrix.shape == (2, 3)


def test_exact_matching_drops_strata_without_overlap() -> None:
    result = pymatchit.matchit(
        "treat ~ sex + region + age",
        make_exact_data(),
        method="exact",
        exact=["sex", "region"],
    )

    matched = result.matched_data()

    assert set(matched.index) == {"a", "b", "c", "d"}
    assert result.weights_.loc["e"] == 0
    assert result.weights_.loc["f"] == 0
    assert pd.isna(result.subclass_.loc["e"])
    assert pd.isna(result.subclass_.loc["f"])


def test_exact_matching_summary_counts_unmatched_units() -> None:
    result = pymatchit.matchit(
        "treat ~ sex + region + age",
        make_exact_data(),
        method="exact",
        exact=["sex", "region"],
    )

    assert result.match_summary() == {
        "method": "exact",
        "estimand": "ATT",
        "n_total": 6,
        "n_matched": 4,
        "n_unmatched": 2,
        "n_subclasses": 2,
        "dropped_strata": 2,
        "ess_treated": 2.0,
        "ess_control": 2.0,
    }


def test_exact_true_uses_all_formula_covariates() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 0, 1, 0],
            "sex": ["F", "F", "M", "M"],
            "region": ["N", "N", "S", "S"],
        }
    )

    result = pymatchit.matchit("treat ~ sex + region", data, method="exact", exact=True)

    assert result.matched_data().shape[0] == 4
    assert result.match_summary()["n_subclasses"] == 2


def test_nearest_matching_without_replacement_uses_each_control_once() -> None:
    result = pymatchit.matchit(
        "treat ~ x",
        make_nearest_data(),
        method="nearest",
        distance="mahalanobis",
        replace=False,
    )

    assert result.matches["treated_index"].tolist() == ["t1", "t2"]
    assert result.matches["control_index"].tolist() == ["c1", "c2"]
    assert result.weights_.to_dict() == {"t1": 1.0, "t2": 1.0, "c1": 1.0, "c2": 1.0}
    assert result.subclass_.loc["t1"] == result.subclass_.loc["c1"]
    assert result.subclass_.loc["t2"] == result.subclass_.loc["c2"]


def test_nearest_matching_with_replacement_reuses_best_control() -> None:
    result = pymatchit.matchit(
        "treat ~ x",
        make_nearest_data(),
        method="nearest",
        distance="mahalanobis",
        replace=True,
    )

    assert result.matches["treated_index"].tolist() == ["t1", "t2"]
    assert result.matches["control_index"].tolist() == ["c1", "c1"]
    assert result.weights_.to_dict() == {"t1": 1.0, "t2": 1.0, "c1": 2.0, "c2": 0.0}
    assert pd.isna(result.subclass_.loc["c2"])


def test_nearest_matching_caliper_excludes_controls() -> None:
    result = pymatchit.matchit(
        "treat ~ x",
        make_nearest_data(),
        method="nearest",
        distance="mahalanobis",
        replace=False,
        caliper=0.05,
    )

    assert result.matches["treated_index"].tolist() == ["t1"]
    assert result.matches["control_index"].tolist() == ["c1"]
    assert result.weights_.to_dict() == {"t1": 1.0, "t2": 0.0, "c1": 1.0, "c2": 0.0}

    tighter = pymatchit.matchit(
        "treat ~ x",
        make_nearest_data(),
        method="nearest",
        distance="mahalanobis",
        replace=False,
        caliper=0.01,
    )

    assert tighter.matches.empty
    assert tighter.weights_.sum() == 0


def test_nearest_matching_respects_exact_blocking() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "sex": ["F", "M", "M", "F"],
            "x": [0.0, 10.0, 0.1, 9.9],
        },
        index=["t_f", "t_m", "c_m", "c_f"],
    )

    result = pymatchit.matchit(
        "treat ~ sex + x",
        data,
        method="nearest",
        distance="mahalanobis",
        exact=["sex"],
        replace=False,
    )

    assert result.matches["treated_index"].tolist() == ["t_f", "t_m"]
    assert result.matches["control_index"].tolist() == ["c_f", "c_m"]


def test_balance_summary_reports_hand_computed_smds() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "x": [2.0, 4.0, 1.0, 1.0],
        }
    )

    result = pymatchit.matchit(
        "treat ~ x",
        data,
        method="nearest",
        distance="mahalanobis",
        replace=False,
    )
    summary = result.balance_summary()

    assert summary.loc["x", "mean_treated_before"] == pytest.approx(3.0)
    assert summary.loc["x", "mean_control_before"] == pytest.approx(1.0)
    assert summary.loc["x", "smd_before"] == pytest.approx(2.0)
    assert summary.loc["x", "mean_treated_after"] == pytest.approx(3.0)
    assert summary.loc["x", "mean_control_after"] == pytest.approx(1.0)
    assert summary.loc["x", "smd_after"] == pytest.approx(2.0)


def test_balance_summary_uses_matching_weights_for_adjusted_smd() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "x": [2.0, 4.0, 2.0, 100.0],
        },
        index=["t1", "t2", "c1", "c2"],
    )

    result = pymatchit.matchit(
        "treat ~ x",
        data,
        method="nearest",
        distance="mahalanobis",
        replace=True,
    )
    summary = result.balance_summary()

    assert result.matches["control_index"].tolist() == ["c1", "c1"]
    assert summary.loc["x", "smd_before"] == pytest.approx(-48.0)
    assert summary.loc["x", "smd_after"] == pytest.approx(1.0)


def test_balance_summary_reports_hand_computed_variance_ratios() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "x": [2.0, 4.0, 1.0, 3.0],
        }
    )

    result = pymatchit.matchit(
        "treat ~ x",
        data,
        method="nearest",
        distance="mahalanobis",
        replace=False,
    )
    summary = result.balance_summary()

    assert summary.loc["x", "smd_before"] == pytest.approx(1.0)
    assert summary.loc["x", "variance_ratio_before"] == pytest.approx(1.0)
    assert summary.loc["x", "variance_ratio_after"] == pytest.approx(1.0)


def test_effective_sample_size_uses_matching_weights() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "x": [2.0, 4.0, 2.0, 100.0],
        },
        index=["t1", "t2", "c1", "c2"],
    )

    result = pymatchit.matchit(
        "treat ~ x",
        data,
        method="nearest",
        distance="mahalanobis",
        replace=True,
    )

    assert result.effective_sample_size() == {
        "treated": 2.0,
        "control": 1.0,
        "total": pytest.approx(2.6666666666666665),
    }


def test_subclassification_att_weights_drop_nonoverlap_bins() -> None:
    treatment = pd.Series([1, 0, 1, 0, 1, 1], index=list("abcdef"))
    propensity = pd.Series([0.1, 0.2, 0.4, 0.5, 0.8, 0.9], index=list("abcdef"))

    result = subclass_match(
        treatment=treatment,
        propensity_score=propensity,
        n_subclasses=3,
        estimand="ATT",
    )

    assert result.subclass.loc[["a", "b", "c", "d"]].to_dict() == {
        "a": 1,
        "b": 1,
        "c": 2,
        "d": 2,
    }
    assert pd.isna(result.subclass.loc["e"])
    assert pd.isna(result.subclass.loc["f"])
    assert result.weights.to_dict() == {
        "a": 1.0,
        "b": 1.0,
        "c": 1.0,
        "d": 1.0,
        "e": 0.0,
        "f": 0.0,
    }
    assert result.dropped_subclasses == 1


def test_subclassification_ate_weights() -> None:
    treatment = pd.Series([1, 1, 0, 1, 0, 0], index=list("abcdef"))
    propensity = pd.Series([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], index=list("abcdef"))

    result = subclass_match(
        treatment=treatment,
        propensity_score=propensity,
        n_subclasses=2,
        estimand="ATE",
    )

    assert result.weights.to_dict() == {
        "a": 1.5,
        "b": 1.5,
        "c": 3.0,
        "d": 3.0,
        "e": 1.5,
        "f": 1.5,
    }
    assert result.dropped_subclasses == 0


def test_matchit_subclass_uses_default_six_subclasses() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
    )

    result = pymatchit.matchit("treat ~ x", data, method="subclass")

    assert result.propensity_score is not None
    assert result.matches is not None
    assert result.diagnostics["n_subclasses_requested"] == 6
    assert result.match_summary()["n_subclasses"] <= 6


def test_matchit_subclass_accepts_user_specified_subclasses_and_ate() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 0, 1, 0, 1, 0, 1, 0],
            "x": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    result = pymatchit.matchit(
        "treat ~ x",
        data,
        method="subclass",
        estimand="ATE",
        n_subclasses=2,
    )

    assert result.estimand == "ATE"
    assert result.diagnostics["n_subclasses_requested"] == 2
    assert result.weights_.sum() > 0


def test_optimal_matching_differs_from_greedy_nearest_on_small_case() -> None:
    treatment = pd.Series([1, 1, 0, 0], index=["t1", "t2", "c1", "c2"])
    distances = np.array(
        [
            [1.0, 2.0],
            [1.0, 100.0],
        ]
    )

    greedy = nearest_neighbor_match(treatment, distances, replace=False)
    optimal = optimal_pair_match(treatment, distances)

    assert greedy.matches["control_index"].tolist() == ["c1", "c2"]
    assert greedy.matches["distance"].sum() == pytest.approx(101.0)
    assert optimal.matches["control_index"].tolist() == ["c2", "c1"]
    assert optimal.matches["distance"].sum() == pytest.approx(3.0)


def test_optimal_matching_caliper_forbids_edges() -> None:
    treatment = pd.Series([1, 1, 0, 0], index=["t1", "t2", "c1", "c2"])
    distances = np.array(
        [
            [1.0, 2.0],
            [1.0, 100.0],
        ]
    )

    result = optimal_pair_match(treatment, distances, caliper=1.5)

    assert result.matches["treated_index"].tolist() == ["t1"]
    assert result.matches["control_index"].tolist() == ["c1"]
    assert result.weights.sum() == 2.0


def test_matchit_optimal_respects_exact_blocking() -> None:
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "sex": ["F", "M", "M", "F"],
            "x": [0.0, 10.0, 0.1, 9.9],
        },
        index=["t_f", "t_m", "c_m", "c_f"],
    )

    result = pymatchit.matchit(
        "treat ~ sex + x",
        data,
        method="optimal",
        distance="mahalanobis",
        exact=["sex"],
    )

    assert result.matches["treated_index"].tolist() == ["t_f", "t_m"]
    assert result.matches["control_index"].tolist() == ["c_f", "c_m"]
    assert result.weights_.to_dict() == {
        "t_f": 1.0,
        "t_m": 1.0,
        "c_m": 1.0,
        "c_f": 1.0,
    }


def test_full_matching_assigns_multiple_controls_to_one_treated() -> None:
    pytest.importorskip("ortools")
    from pymatchit.matching.full import full_match

    treatment = pd.Series([1, 0, 0], index=["t1", "c1", "c2"])
    distances = np.array([[1.0, 3.0]])

    result = full_match(treatment, distances)

    assert result.matches.loc[0, "n_treated"] == 1
    assert result.matches.loc[0, "n_control"] == 2
    assert result.matches.loc[0, "total_distance"] == pytest.approx(4.0)
    assert result.weights.to_dict() == {"t1": 1.0, "c1": 0.5, "c2": 0.5}
    assert result.subclass.to_dict() == {"t1": 1, "c1": 1, "c2": 1}


def test_full_matching_caliper_drops_forbidden_units() -> None:
    pytest.importorskip("ortools")
    from pymatchit.matching.full import full_match

    treatment = pd.Series([1, 0, 0], index=["t1", "c1", "c2"])
    distances = np.array([[1.0, 3.0]])

    result = full_match(treatment, distances, caliper=2.0)

    assert result.matches.loc[0, "n_control"] == 1
    assert result.weights.to_dict() == {"t1": 1.0, "c1": 1.0, "c2": 0.0}
    assert pd.isna(result.subclass.loc["c2"])
    assert result.dropped_units == 1


def test_matchit_full_is_experimental_and_respects_exact_blocks() -> None:
    pytest.importorskip("ortools")
    data = pd.DataFrame(
        {
            "treat": [1, 1, 0, 0],
            "sex": ["F", "M", "F", "M"],
            "x": [0.0, 10.0, 0.1, 10.1],
        },
        index=["t_f", "t_m", "c_f", "c_m"],
    )

    with pytest.warns(UserWarning, match="experimental"):
        result = pymatchit.matchit(
            "treat ~ sex + x",
            data,
            method="full",
            distance="mahalanobis",
            exact=["sex"],
        )

    assert result.diagnostics["experimental"] is True
    assert result.weights_.to_dict() == {
        "t_f": 1.0,
        "t_m": 1.0,
        "c_f": 1.0,
        "c_m": 1.0,
    }
    assert result.match_summary()["n_subclasses"] == 2


def test_love_plot_runs_with_agg_backend() -> None:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes

    result = pymatchit.matchit(
        "treat ~ x + z",
        make_plot_data(),
        method="nearest",
        distance="mahalanobis",
    )

    ax = pymatchit.love_plot(result)

    assert isinstance(ax, Axes)
    plt.close(ax.figure)


def test_ecdf_qq_jitter_and_histogram_plots_run_with_agg_backend() -> None:
    from matplotlib import pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    result = pymatchit.matchit(
        "treat ~ x + z",
        make_plot_data(),
        method="nearest",
        distance="logit",
    )

    figures = [
        pymatchit.ecdf_plot(result, "x"),
        pymatchit.qq_plot(result, "x"),
        pymatchit.histogram_plot(result),
    ]
    jitter_ax = pymatchit.jitter_plot(result)

    for figure in figures:
        assert isinstance(figure, Figure)
        plt.close(figure)
    assert isinstance(jitter_ax, Axes)
    plt.close(jitter_ax.figure)


def test_density_plot_runs_for_continuous_and_categorical_covariates() -> None:
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure

    result = pymatchit.matchit(
        "treat ~ x + z + sex",
        make_plot_data(),
        method="nearest",
        distance="mahalanobis",
    )

    continuous = pymatchit.density_plot(result, "x")
    categorical = pymatchit.density_plot(result, "sex[T.M]")

    assert isinstance(continuous, Figure)
    assert isinstance(categorical, Figure)
    plt.close(continuous)
    plt.close(categorical)
