"""Small Lalonde-like design-stage matching demo."""

import numpy as np
import pandas as pd

import pymatchit


def make_lalonde_like_data() -> pd.DataFrame:
    """Create a tiny observational dataset with Lalonde-style covariates."""
    return pd.DataFrame(
        {
            "treat": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "age": [25, 32, 41, 29, 23, 35, 39, 28, 45, 31],
            "educ": [10, 12, 11, 9, 10, 12, 14, 9, 13, 11],
            "black": [1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
            "hispan": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "married": [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            "nodegree": [1, 0, 1, 1, 1, 0, 0, 1, 0, 1],
            "re74": [0, 15_000, 8_000, 2_000, 0, 18_000, 22_000, 3_000, 30_000, 4_000],
            "re75": [
                0,
                12_000,
                9_000,
                1_000,
                500,
                14_000,
                19_000,
                2_000,
                25_000,
                3_000,
            ],
            "re78": [
                16_000,
                21_000,
                18_000,
                14_000,
                9_000,
                20_000,
                24_000,
                8_000,
                28_000,
                10_000,
            ],
        }
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Simple weighted mean for the separate outcome-analysis step."""
    return float(np.average(values, weights=weights))


def main() -> None:
    data = make_lalonde_like_data()
    formula = "treat ~ age + educ + black + hispan + married + nodegree + re74 + re75"

    fit = pymatchit.matchit(
        formula,
        data,
        method="nearest",
        distance="logit",
        replace=False,
        caliper=1.5,
    )

    print("Matched rows")
    print(fit.matched_data())
    print()

    print("Balance summary")
    print(fit.balance_summary().round(3))
    print()

    print("Match summary")
    print(fit.match_summary())
    print()

    matched = fit.matched_data()
    treated = matched["treat"] == 1
    treated_mean = weighted_mean(
        matched.loc[treated, "re78"],
        matched.loc[treated, "weights"],
    )
    control_mean = weighted_mean(
        matched.loc[~treated, "re78"],
        matched.loc[~treated, "weights"],
    )

    print("Separate outcome-analysis example")
    print(f"Weighted treated mean re78: {treated_mean:.2f}")
    print(f"Weighted control mean re78: {control_mean:.2f}")
    print(f"Difference in weighted means: {treated_mean - control_mean:.2f}")


if __name__ == "__main__":
    main()
