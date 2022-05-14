import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

symbols = np.array(["circle", "x"])


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def plot_adaboost(adaBoost, test_X, test_y, T, lims):
    title = "Adaboost"

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$ Adaboost\ with\ {{{m}}} \
    learners$" for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        def pred(X):
            return adaBoost.partial_predict(X, t)

        fig.add_traces(
            [decision_surface(pred, lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y, symbol=symbols[
                            test_y.astype(int)],
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig.update_layout(
        title=rf"$\textbf{{ Decision Boundaries to num of Learners - {title}}}$"
        , margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()
    # fig.write_image('Q2_adaboost.png')


def Q_2(n_learners, test_X, test_y, train_X, train_y):
    training_errors, test_errors = [], []
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)
    ensemble = np.arange(1, n_learners)
    for l in ensemble:
        training_errors.append(adaBoost.partial_loss(train_X, train_y, l))
        test_errors.append(adaBoost.partial_loss(test_X, test_y, l))
    plt.plot(ensemble, training_errors, label='Train errors', markersize=3)
    plt.plot(ensemble, test_errors, label='Test errors')
    plt.title('Adaboost Algorithm error as function of iterations')
    plt.legend()
    plt.show()
    return adaBoost, test_errors


def Q_3(adaBoost, best_num_of_learners, lims, test_X, test_y):
    def pred(X):
        return adaBoost.partial_predict(X, best_num_of_learners)

    accuracy = 1 - adaBoost.partial_loss(test_X, test_y, best_num_of_learners)
    fig = go.Figure([decision_surface(pred, lims[0], lims[1], showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                                showlegend=False,
                                marker=dict(color=test_y, symbol=symbols[
                                    test_y.astype(int)],
                                            colorscale=[custom[0],
                                                        custom[-1]],
                                            line=dict(color="black",
                                                      width=1)))])
    fig.update_layout(
        title=rf"$\textbf{{ Best ensemble decision boundary:  "
              rf"{best_num_of_learners}   Accuracy: {accuracy}}}$"
        , margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def Q4(adaBoost, D_t, lims, test_y, train_X, best_num_of_learners):
    def pred(X):
        return adaBoost.partial_predict(X, best_num_of_learners)

    fig = go.Figure([decision_surface(pred, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                mode="markers",
                                showlegend=False,
                                marker=dict(color=test_y, symbol=symbols[
                                    test_y.astype(int)], size=D_t,
                                            colorscale=[custom[0],
                                                        custom[-1]],
                                            line=dict(color="black",
                                                      width=1)))])
    fig.update_layout(
        title=rf"$\textbf{{ Decision surface: training set with weighted samples }}$"
        , margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost, test_errors = Q_2(n_learners, test_X, test_y, train_X, train_y)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    plot_adaboost(adaBoost, test_X, test_y, T, lims)

    # Question 3: Decision surface of best performing ensemble
    best_num_of_learners = np.min(np.argmin(test_errors)) + 1
    Q_3(adaBoost, best_num_of_learners, lims, test_X, test_y)

    # Question 4: Decision surface with weighted samples
    D_t = adaBoost.D_
    D_t = D_t / np.max(D_t) * 5
    Q4(adaBoost, D_t, lims, test_y, train_X, best_num_of_learners)

    # Question 5: graphs as in Q1, Q4
    noise_n = 0.4
    (train_X_n, train_y_n), (test_X_n, test_y_n) = generate_data(train_size,
                                   noise_n), generate_data(test_size, noise_n)
    adaBoost_n, test_errors_n = Q_2(n_learners, test_X_n, test_y_n, train_X_n,
                                    train_y_n)
    best_num_of_learners_n = np.min(np.argmin(test_errors_n)) + 1
    D_t_n = adaBoost_n.D_
    D_t_n = D_t_n / np.max(D_t_n) * 5

    lims_n = np.array([np.r_[train_X_n, test_X_n].min(axis=0),
                     np.r_[train_X_n, test_X_n].max(axis=0)]).T + np.array(
        [-.1, .1])

    Q4(adaBoost_n, D_t_n, lims_n, test_y_n, train_X_n, n_learners)


if __name__ == '__main__':
    np.random.seed(0)
    noise = 0
    fit_and_evaluate_adaboost(noise)
