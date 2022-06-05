from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from typing import Tuple
from IMLearn.metrics.loss_functions import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    min_val, max_val = -1.2, 2
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(min_val, max_val, n_samples)
    y = response(X)

    epsilon = np.random.normal(scale=noise, size=n_samples)
    y_noise = y + epsilon

    train_X, train_y, test_X, test_y = split_train_test(
        pd.DataFrame(X), pd.Series(y_noise), (2 / 3))
    train_X, train_y, test_X, test_y = train_X.to_numpy().reshape((-1)), \
                                       train_y.to_numpy().reshape(
                                           (-1)), test_X.to_numpy().reshape(
        (-1)), test_y.to_numpy().reshape((-1))
    plt.figure()
    plt.plot(train_X, train_y, 'o')
    plt.plot(test_X, test_y, '*')
    plt.plot(X, y)
    plt.title("noise= " + str(noise) + " model and scattered train & test "
                                       "with n_samples: " + str(n_samples))
    plt.legend(['Train', 'Test', 'True Model'])
    plt.show()
    # plt.savefig('Ex5_'+' noise= ' + str(noise) + ', n_samples= ' + str(
    # n_samples) + '.png')

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    max_degree = 10
    cv = 5

    avg_train_err = []
    avg_validation_err = []

    K = np.arange(11)
    for i in K:
        pf = PolynomialFitting(i)
        train_score, validation_score = cross_validate(pf, train_X, train_y,
                                                       mean_square_error, cv)
        avg_train_err.append(train_score)
        avg_validation_err.append(validation_score)

    plt.figure()
    plt.plot(np.arange(max_degree + 1), avg_train_err)
    plt.plot(np.arange(max_degree + 1), avg_validation_err)
    plt.title('Train and Validation loss as function of polynomial degree')
    plt.legend(['train', 'validation'])
    plt.show()
    # plt.savefig('Ex5_q2'+' noise= ' + str(noise) + ', n_samples= ' + str(
    #     n_samples) +'.png')

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(avg_validation_err)
    pf_best_deg = PolynomialFitting(int(best_k))
    pf_best_deg.fit(train_X, train_y)
    score = round(pf_best_deg.loss(test_X, test_y), 2)
    print("The best K is: " + str(best_k))
    print("The Validation error: " + str(
        round(avg_validation_err[int(best_k)], 2)) + " The test error: " + str(
        score))


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:50], y[:50]
    test_X, test_y = X[50:], y[50:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    n_evaluations = 500
    cv = 5
    lamdas = np.linspace(0.001, 5, n_evaluations).astype(float)

    avg_train_err_ridge, avg_validation_err_ridge = [], []
    avg_train_err_lasso, avg_validation_err_lasso = [], []

    for lam in lamdas:
        rr = RidgeRegression(lam)
        train_score_ridge, validation_score_ridge = cross_validate(rr, train_X,
                                                                   train_y,
                                                                   mean_square_error,
                                                                   cv=cv)
        avg_train_err_ridge.append(train_score_ridge)
        avg_validation_err_ridge.append(validation_score_ridge)

        lr = Lasso(lam)
        train_score_lasso, validation_score_lasso = cross_validate(lr, train_X,
                                                                   train_y,
                                                                   mean_square_error,
                                                                   cv=cv)
        avg_train_err_lasso.append(train_score_lasso)
        avg_validation_err_lasso.append(validation_score_lasso)

    plt.figure()
    plt.plot(lamdas, avg_train_err_ridge)
    plt.plot(lamdas, avg_validation_err_ridge)
    plt.plot(lamdas, avg_train_err_lasso)
    plt.plot(lamdas, avg_validation_err_lasso)
    plt.title('Train and Validation loss as function of polynomial degree')
    plt.legend(['avg_train_err_ridge', 'avg_validation_err_ridge',
                'avg_train_err_lasso', 'avg_validation_err_lasso'])
    plt.show()
    # plt.savefig('Ex5_q7.png')

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lam_ind = np.argmin(avg_validation_err_ridge)
    ridge_best_lam = lamdas[ridge_best_lam_ind]

    lasso_best_lam_ind = np.argmin(avg_validation_err_lasso)
    lasso_best_lam = lamdas[lasso_best_lam_ind]

    # print('ridge best lambda: ' + str(ridge_best_lam))
    # print('lasso best lambda: ' + str(lasso_best_lam))
    # print('linear regression best lambda: ' + str(lr_best_lam))

    ridge = RidgeRegression(ridge_best_lam)
    lasso = Lasso(lasso_best_lam)
    lr = LinearRegression()

    ridge.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    lr.fit(train_X, train_y)

    score_ridge = round(ridge.loss(test_X, test_y), 2)
    score_lasso = round(mean_square_error(test_y, lasso.predict(test_X)), 2)
    score_lr = round(lr.loss(test_X, test_y), 2)

    # print("The Validation error of ridge: " + str(score_ridge))
    # print("The Validation error of lasso: " + str(score_lasso))
    # print("The Validation error of linear regression: " + str(score_lr))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)  # for Q4
    select_polynomial_degree(n_samples=1500, noise=10)  # for Q5
    select_regularization_parameter()
