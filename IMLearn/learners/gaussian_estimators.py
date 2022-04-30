from __future__ import annotations
import numpy as np
import numpy.random
from numpy.linalg import inv, det, slogdet
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean sssn variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.mu_, self.var_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        samples_length = numpy.size(X)
        self.mu_ = np.mean(X)

        denominator = samples_length - 1
        if self.biased_:  # using the biased variance estimator from class
            denominator = samples_length
        self.var_ = (1 / denominator) * np.sum(numpy.power((X - self.mu_), 2))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        # pdfs = np.empty
        # pdf_val = None
        # for i in range(X.size):
        #     pdf_i = (1 / np.sqrt(2 * np.pi * self.var_)) * \
        #        (np.exp(-(X[i] - self.mu_) ** 2 / (2 * self.var_)))
        #     np.append(pdfs, pdf_i)

        pdfs = (1 / np.sqrt(2 * np.pi * self.var_)) * \
               (np.exp(-(X - self.mu_) ** 2 / (2 * self.var_)))
        return pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        ll = -np.sum(np.log(2 * np.pi * (sigma ** 2)) / 2 + ((X - mu) ** 2) / \
                     (2 * (sigma ** 2)))

        return ll


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features )
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X.T)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features )
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")

        d = np.ma.size(X)  # d is the dimension of X
        pdfs = (1. / np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov_))) * (
            np.exp(-(np.linalg.solve(self.cov_, (X - self.mu_)).T.dot(X -
                                                                      self.mu_)) / 2))

        return pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: (np.ndarray, np.ndarray),
                       X: (np.ndarray, np.ndarray)) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        d = np.shape(X)[0]  # number of features
        m = np.shape(X)[1]  # number of samples

        inv_cov = np.linalg.inv(cov)
        ll_sum = 0
        for x in X:  # calculating the loglikelihood sum component
            ll_sum += (x - mu) @ inv_cov @ (x - mu).T

        ll = -0.5 * (d * m * np.log(2 * np.pi) + d * np.log(np.linalg.det(
            cov)) + ll_sum)
        return ll
