from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from ..base import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = [], []

    folds = np.remainder(np.arange(X.shape[0]), cv)
    for k in range(cv):
        train_k_x = X[folds != k]
        train_k_y = y[folds != k]
        validate_k_x = X[folds == k]
        validate_k_y = y[folds == k]

        estimator.fit(train_k_x, train_k_y)
        S_k_pred = estimator.predict(train_k_x)
        V_k_pred = estimator.predict(validate_k_x)

        S_loss_d = scoring(train_k_y, S_k_pred)
        V_loss_d = scoring(validate_k_y, V_k_pred)

        train_score.append(S_loss_d)
        validation_score.append(V_loss_d)

    return np.average(train_score), np.average(validation_score)

