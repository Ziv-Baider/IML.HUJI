from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        pi, vars = [], []
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((self.classes_.size, X.shape[1]))
        self.vars_ = np.zeros(self.mu_.shape)  # (n_classes, n_features)
        for i, clas in enumerate(self.classes_):
            # calculate pi
            pi.append((y == clas).mean())

            # calculate mean
            indexes = [j for j, y_j in enumerate(y) if y[j] == clas]
            X_clas = np.array([X[j] for j in indexes])
            self.mu_[i] = np.mean(X_clas, axis=0)
            vars.append(np.var(X_clas))
            # calculate variance
            for f in range(X_clas.shape[1]):
                self.vars_[i][f] = np.var(X_clas[:, f], ddof=1)
        self.pi_ = np.array(pi)
        self.mu_ = self.mu_.T
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
            responses[i_ll] = self.classes_[np.argmax(x_ll)]
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
        n_samples, n_features = X.shape[0], X.shape[1]
        likelihood = np.zeros((n_samples, self.classes_.size))

        for j, class_j in enumerate(self.classes_):
            var = self.vars_[j] * np.identity((self.vars_[j].reshape(-1, 1)).shape[0])
            var_inv = np.linalg.inv(var)
            l = self.pi_[j] * np.sqrt(1 / ((2 * np.pi) ** n_features *
                                           np.linalg.det(var))) * np.exp(
                -1 / 2 * np.diag((X - self.mu_.T[
                    j]) @ var_inv @ (X - self.mu_.T[j]).T))
            likelihood[:, j] = l

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
