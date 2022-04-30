from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from IMLearn.metrics.loss_functions import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        pi = []
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.cov_ = np.zeros((X.shape[1],  X.shape[1]))
        for i, clas in enumerate(self.classes_):
            # calculate pi
            pi.append((y == clas).mean())

            # calculate mean
            indexes = [j for j, y_j in enumerate(y) if y[j] == clas]
            X_clas = np.array([X[j] for j in indexes])
            self.mu_[i] = np.mean(X_clas, axis=0)
            # calculate covariance
            self.cov_ += (X_clas - self.mu_[i, :]).T @ (X_clas - self.mu_[i, :])

        self.pi_ = np.array(pi)
        self.mu_ = self.mu_.T
        # self.cov_ = np.cov(X.T) / (y.size - self.classes_.size)  # unbiased
        self.cov_ /= (y.size - self.classes_.size)  # unbiased

        # self.cov_ = np.cov(X.T)

        self._cov_inv = np.linalg.inv(self.cov_)

        self.fitted_ = True

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
        responses = np.zeros(X.shape[0])
        ll = self.likelihood(X)
        for i_ll, x_ll in enumerate(ll):
            responses[i_ll] =self.classes_[np.argmax(x_ll)]
        return responses.T

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        likelihood = np.zeros((X.shape[0], self.classes_.size))
        for i, x_i in enumerate(X):
            for j, class_j in enumerate(self.classes_):
                ll = ((x_i @ self._cov_inv @ self.mu_[:,j]) + (-0.5 *
                    self.mu_[:,j].T @ self._cov_inv @ self.mu_[:,j] + np.log(self.pi_[j])))
                likelihood[i][j] = ll

        return likelihood

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
