from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    arr = np.load(filename)

    return arr[:, :2], arr[:, 2].astype(int)


def plot_losses(losses, iterations, n, f):
    title = "Loss of Perceptron a.f of iterations for dataset: " + n
    xaxis_title = "Number of iterations"
    yaxis_title = "Loss"

    fig = px.line(x=iterations, y=losses, title=title)
    fig.update_xaxes(title_text=xaxis_title)
    fig.update_yaxes(title_text=yaxis_title)
    fig.show()
    # fig.write_image("Loss of Perceptron a.f of iterations.%s.png" % n)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        losses = []

        def callback(perceptron, X_arg, y_arg):
            loss = perceptron.loss(X, y)
            losses.append(loss)
            return loss

        # Fit Perceptron and record loss in each fit iteration
        p = Perceptron(callback=callback).fit(X, y)
        iterations = np.arange(0, len(losses), 1)

        plot_losses(np.array(losses), iterations, n, f)


def plot_gaussian_classifiers(f, gnb, lda, X, y, pred_lda, pred_gnb):
    gnb_acc = accuracy(y, pred_gnb)
    lda_acc = accuracy(y, pred_lda)
    df = pd.DataFrame()
    symbols = np.array(["circle", "diamond", "triangle-up"])

    df["x"], df["y"] = X[:, 0], X[:, 1]
    df["label"] = y
    df["pred_gnb"], df["pred_lda"] = pred_gnb, pred_lda

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Gaussian Naive Bayes. Accuracy:     " + str(gnb_acc),
        "LDA. Accuracy:     " + str(lda_acc)))
    # plot GNB
    fig.add_trace(
        go.Scatter(x=df["x"], y=df["y"], mode="markers",
                   marker=dict(color=df["pred_gnb"],
                               symbol=symbols[df["label"].astype(int)]),
                   showlegend=False),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=gnb.mu_.T[:, 0], y=gnb.mu_.T[:, 1],
                             mode="markers",
                             marker=dict(color="black", symbol="x"),
                             showlegend=False), row=1,
                  col=1)
    for i in range(np.shape(gnb.mu_.T)[0]):
        fig.add_trace(get_ellipse(gnb.mu_.T[i], gnb.vars_[i] * np.identity(
            np.shape(gnb.vars_[i].reshape(-1, 1))[0])),
                      row=1, col=1)
    # plot LDA
    fig.add_trace(
        go.Scatter(x=df["x"], y=df["y"], mode="markers",
                   marker=dict(color=df["pred_lda"],
                               symbol=symbols[df["label"].astype(int)]),
                   showlegend=False),
        row=1, col=2)
    fig.add_trace(
        go.Scatter(x=lda.mu_.T[:, 0], y=lda.mu_.T[:, 1], mode="markers",
                   marker=dict(color="black", symbol="x"),
                   showlegend=False), row=1, col=2)
    for i in range(np.shape(lda.classes_)[0]):
        fig.add_trace(get_ellipse(lda.mu_.T[i, :], lda.cov_), row=1, col=2)
    fig.update_layout(title=rf"$\textbf{{ {f} Dataset}}$")
    fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        gnb = GaussianNaiveBayes().fit(X, y)

        lda_prediction = lda.predict(X)
        gnb_prediction = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy

        plot_gaussian_classifiers(f, gnb, lda, X, y, lda_prediction,
                                  gnb_prediction)

def quiz():
    #1
    # y = np.array([0, 0, 1, 1, 1, 1, 2, 2]).T
    # X = np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(1, -1).T

    #2
    y = np.array([0, 0, 1, 1, 1, 1]).T
    X = np.array([[1,1], [1,2], [2,3], [2,4], [3,3], [3,4]]).reshape(1, -1).T

    #3
    # X = np.array([[1,1], [1,2], [2,3], [2,4], [3,3], [3,4]]).reshape(1, -1).T

    #4


    lda = LDA().fit(X, y)
    gnb = GaussianNaiveBayes().fit(X, y)
    # print(gnb.pi_)
    # print(gnb.mu_)
    print(gnb.vars_)


if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    quiz()