from __future__ import annotations
from typing import NoReturn, Tuple
from ...base import BaseEstimator
import numpy as np
from itertools import product
from IMLearn.metrics.loss_functions import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        n_samples, n_features = X.shape[0], X.shape[1]
        best_loss = np.inf

        # Iterating over all samples of X's features with sign in (1, -1)
        for sign, j in product([-1, 1], range(n_features)):
            threshold, loss = self._find_threshold(X[:, j], y, sign)
            if loss < best_loss:  # if fe found a smaller loss
                best_loss = loss
                self.threshold_ = threshold
                self.j_ = j
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # return sign if above threshold and -sign otherwise
        return self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # Sort the values and labels for efficiency
        sorted_indexes = np.argsort(values)
        sorted_values = values[sorted_indexes]
        sorted_labels = np.sign(labels[sorted_indexes])

        # Initiate D
        D = np.abs(labels)
        D /= np.sum(D)
        sorted_D = D[sorted_indexes]

        # Initiate Threshold
        thresholds = np.concatenate(
            [[-np.inf], (sorted_values[1:] + sorted_values[:-1]) / 2, [np.inf]])
        # find the losses
        min_thr_loss = np.sum(sorted_D[sorted_labels == sign])
        thresholds_losses = np.append(min_thr_loss, min_thr_loss - np.cumsum(
            sorted_D * sorted_labels * sign))
        # fins min index for min loss and threshold
        min_loss_idx = np.argmin(thresholds_losses)
        thr, thr_err = thresholds[min_loss_idx], thresholds_losses[min_loss_idx]
        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
