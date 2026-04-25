"""Distance and propensity-score helpers for future matching methods."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression


SUPPORTED_DISTANCES = {"propensity", "logit", "mahalanobis"}


def estimate_propensity_score(
    treatment: pd.Series,
    covariates: pd.DataFrame,
) -> pd.Series:
    """Estimate propensity scores with logistic regression."""
    _validate_covariates(covariates)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(covariates.to_numpy(), treatment.to_numpy())
    scores = model.predict_proba(covariates.to_numpy())[:, 1]
    return pd.Series(scores, index=covariates.index, name="propensity_score")


def compute_distance_matrix(
    treatment: pd.Series,
    covariates: pd.DataFrame,
    distance: str,
    propensity_score: pd.Series | None = None,
) -> np.ndarray:
    """Compute a treated-by-control distance matrix for ATT matching."""
    if distance not in SUPPORTED_DISTANCES:
        supported = ", ".join(sorted(SUPPORTED_DISTANCES))
        raise ValueError(f"distance must be one of: {supported}.")

    treated_mask = treatment.astype(int).to_numpy() == 1
    control_mask = ~treated_mask

    if distance == "propensity":
        if propensity_score is None:
            propensity_score = estimate_propensity_score(treatment, covariates)
        return propensity_score_distance(propensity_score, treated_mask, control_mask)

    if distance == "logit":
        if propensity_score is None:
            propensity_score = estimate_propensity_score(treatment, covariates)
        return logit_propensity_distance(propensity_score, treated_mask, control_mask)

    return mahalanobis_distance(covariates, treated_mask, control_mask)


def propensity_score_distance(
    propensity_score: pd.Series,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
) -> np.ndarray:
    """Pairwise absolute distance between treated and control propensity scores."""
    scores = propensity_score.to_numpy()
    return np.abs(scores[treated_mask, None] - scores[control_mask][None, :])


def logit_propensity_distance(
    propensity_score: pd.Series,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
) -> np.ndarray:
    """Pairwise absolute distance between logits of propensity scores."""
    scores = np.clip(propensity_score.to_numpy(), 1e-8, 1 - 1e-8)
    logits = np.log(scores / (1 - scores))
    return np.abs(logits[treated_mask, None] - logits[control_mask][None, :])


def mahalanobis_distance(
    covariates: pd.DataFrame,
    treated_mask: np.ndarray,
    control_mask: np.ndarray,
) -> np.ndarray:
    """Pairwise Mahalanobis distance between treated and control covariates."""
    _validate_covariates(covariates)
    x = covariates.to_numpy()
    covariance = np.cov(x, rowvar=False)
    covariance = np.atleast_2d(covariance)
    inverse_covariance = np.linalg.pinv(covariance)
    return cdist(
        x[treated_mask],
        x[control_mask],
        metric="mahalanobis",
        VI=inverse_covariance,
    )


def _validate_covariates(covariates: pd.DataFrame) -> None:
    if covariates.isna().any().any():
        raise ValueError("covariates must not contain NaNs.")
    non_numeric = [
        column
        for column in covariates.columns
        if not pd.api.types.is_numeric_dtype(covariates[column])
    ]
    if non_numeric:
        columns = ", ".join(non_numeric)
        raise ValueError(f"covariates must be numeric; non-numeric columns: {columns}.")
    zero_variance = covariates.columns[covariates.var(axis=0) == 0].tolist()
    if zero_variance:
        columns = ", ".join(zero_variance)
        raise ValueError(f"covariates must have nonzero variance: {columns}.")
