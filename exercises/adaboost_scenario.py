import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
# from utils import decision_surface
from matplotlib.colors import ListedColormap


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


def decision_boundaries(classifier, X, y, num_classifiers=1, weights=None):
    """
    Plot the decision boundaries of a binary classfiers over X \subseteq R^2

    Parameters
    ----------
    classifier : a binary classifier, implements classifier.predict(X)
    X : m*2 matrix whose rows correspond to the data points
    y : m dimensional vector of binary labels
    title_str : optional title
    weights : weights for plotting X
    """
    cm = ListedColormap(['#AAAAFF','#FFAAAA'])
    cm_bright = ListedColormap(['#0000FF','#FF0000'])
    h = .003  # step size in the mesh
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()], num_classifiers)
    Z = classifier.partial_predict(np.c_[xx.ravel(), yy.ravel()], num_classifiers)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=cm)
    # Plot also the training points
    if weights is not None:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=weights, cmap=cm_bright)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks([])
    plt.yticks([])
    plt.title(f'num classifiers = {num_classifiers}')
    plt.draw()


def plot_adaboost(adaBoost, test_X, test_y, T, lims):
    symbols = np.array(["circle", "x"])
    title = "Adaboost Algorithm"

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$ Adaboost with \textbf{{{ m  }}}$"
                                        for m in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        def pred(X):
            return adaBoost.partial_predict(X, t)

        fig.add_traces([decision_surface(pred,lims[0], lims[1],showscale=False),
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
        title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    # fig.show() #todo
    fig.write_image('Q2_adaboost.png')


def plot_errors_afo_learners(n_learners, test_X, test_y, train_X, train_y):
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
    plt.show()  #todo: return it
    return adaBoost, test_errors


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaBoost, test_errors = plot_errors_afo_learners(n_learners, test_X,
                                                     test_y, train_X, train_y)


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    # lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

    plot_adaboost(adaBoost, test_X, test_y, T, lims)




    # Question 3: Decision surface of best performing ensemble
    # lower_test_error_ind = np.min(np.argmin(test_errors))
    lower_test_error_ind = np.min(np.argmin(test_errors)) +1
    print(lower_test_error_ind)
    best_ensemble_size = test_errors[lower_test_error_ind]
    # decidion_boundaries_new(adaBoost, test_X, test_y, lower_test_error_ind)
    # decision_boundaries(adaBoost, test_X, test_y, lower_test_error_ind)
    # plt.show()  # todo

    # Question 4: Decision surface with weighted samples
    D_t = adaBoost.D_
    normalized_D_t = D_t / np.max(D_t) * 5

    # Question 5: graphs as in Q1, Q4
    noise_n = 0.4
    (train_X_n, train_y_n), (test_X_n, test_y_n) = generate_data(train_size,
                                    noise_n), generate_data( test_size, noise)
    # adaBoost, test_errors = plot_errors_afo_learners(n_learners, test_X,
    #                                                  test_y, train_X, train_y)

if __name__ == '__main__':
    np.random.seed(0)
    noise = 0
    fit_and_evaluate_adaboost(noise)
