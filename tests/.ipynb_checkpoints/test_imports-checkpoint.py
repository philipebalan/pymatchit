import pandas as pd

import pymatchit


def make_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treat": [1, 0, 1, 0, 0],
            "age": [31, 42, 29, 55, 48],
            "income": [70_000, 62_000, 80_000, 59_000, 64_000],
            "x1": [0.2, -1.1, 1.4, 0.0, -0.4],
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

    assert result.matches is None
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
