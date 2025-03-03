from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
import numpy as np


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        self.linear_reg_model = LinearRegression(False)
        self.k = k

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        X_transformed = self.__transform(X)
        self.linear_reg_model.fit(X_transformed, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.linear_reg_model.predict(self.__transform(X))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return self.linear_reg_model.loss(self.__transform(X), y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, self.k + 1)



############################################

# from _future_ import annotations
# from typing import NoReturn
# from . import LinearRegression
# from ...base import BaseEstimator
# import numpy as np

#
# class PolynomialFitting(BaseEstimator):
#     """
#     Polynomial Fitting using Least Squares estimation
#     """
#
#     def __init__(self, k: int) -> PolynomialFitting:
#         """
#         Instantiate a polynomial fitting estimator
#
#         Parameters
#         ----------
#         k : int
#             Degree of polynomial to fit
#         """
#         super().__init__()
#         self.lr_ = LinearRegression(False)
#         self.k_ = k
#
#     def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
#         """
#         Fit Least Squares model to polynomial transformed samples
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to fit an estimator for
#
#         y : ndarray of shape (n_samples, )
#             Responses of input data to fit to
#         """
#         X_vander = self.__transform(X)
#         self.lr_.fit(X_vander, y)
#
#     def _predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Predict responses for given samples using fitted estimator
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Input data to predict responses for
#
#         Returns
#         -------
#         responses : ndarray of shape (n_samples, )
#             Predicted responses of given samples
#         """
#         X_vander = self.__transform(X)
#         return self.lr_.predict(X_vander)
#
#     def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
#         """
#         Evaluate performance under MSE loss function
#
#         Parameters
#         ----------
#         X : ndarray of shape (n_samples, n_features)
#             Test samples
#
#         y : ndarray of shape (n_samples, )
#             True labels of test samples
#
#         Returns
#         -------
#         loss : float
#             Performance under MSE loss function
#         """
#         X_vander = self.__transform(X)
#         return self.lr_.loss(X_vander, y)
#
#     def __transform(self, X: np.ndarray) -> np.ndarray:
#         """
#         Transform given input according to the univariate polynomial transformation
#
#         Parameters
#         ----------
#         X: ndarray of shape (n_samples,)
#
#         Returns
#         -------
#         transformed: ndarray of shape (n_samples, k+1)
#             Vandermonde matrix of given samples up to degree k
#         """
#         return np.vander(X, self.k_ + 1, True)