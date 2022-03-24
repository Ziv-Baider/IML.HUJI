from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"
import plotly.figure_factory as ff


def plot_abs_dist(x_axis, esimated_expectation, true_expectation):
    abs_dist = []

    for i in range(np.size(true_expectation)):
        abs_dist.append(np.abs(esimated_expectation[i] - true_expectation[i]))

    fig = make_subplots(rows=1, cols=2) \
        .add_traces([go.Scatter(x=x_axis, y=abs_dist, mode='markers',
                                marker=dict(color="blue"), showlegend=False)],
                    rows=[1], cols=[1]) \
        .update_layout(title_text=r"$\text{("r"1) Absolute distance between "
                                  r"estimated and true value expectaion as "
                                  r"function of sample size}$",
                       xaxis_title="Number of Samples", yaxis_title="Abs "
                                                                    "Dist")
    fig.show()


def plot_pdf(x_axis, pdfs):
    make_subplots(rows=1, cols=1) \
        .add_traces([go.Scatter(x=x_axis, y=pdfs, mode='markers',
                                marker=dict(color="blue"), showlegend=False)],
                    rows=[1], cols=[1]) \
        .update_layout(title_text=r"$\text{(2) Empirical pdf as function of "
                                  r"ordered samples}$", xaxis_title="Samples",
                       yaxis_title="Power Density Function ").show()


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mean, var, samplea_num = 10, 1, 1000
    uv = UnivariateGaussian()
    samples = np.random.normal(mean, var, samplea_num)
    fitted = uv.fit(samples)
    print("univariate gaussian mean and var are:")
    print(uv.mu_, uv.var_)

    # Question 2 - Empirically showing sample mean is consistent
    i, j, size = 0, 10, 10
    esimated_expectation, true_expectation, x_axis, pdfs = [], [], [], []

    while j < samplea_num:
        fit_i = uv.fit(samples[i:j])
        esimated_expectation.append(fit_i.mu_)
        true_expectation.append(mean)
        x_axis.append((j - i))
        j += 10

    plot_abs_dist(x_axis, esimated_expectation, true_expectation)

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = (fitted.pdf(samples))
    plot_pdf(samples, pdfs)


def heatmap_plot(z, x, y):
    go.Figure(go.Heatmap(x=x, y=y, z=z),
              layout=go.Layout(title='Heatmap of loglikelihood of f1 and f3 '
                                     'values', height=500,
                               width=500)).update_layout(
        xaxis_title='f3', yaxis_title='f1').show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    samplea_num = 1000
    mv = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0],
                    [0.5, 0, 0, 1]]).reshape(4, 4)
    samples = np.random.multivariate_normal(mu, cov, samplea_num)
    mv_fitted = mv.fit(samples)
    print("multivariate gaussian mean is: " , mv.mu_)
    print("multivariate gaussian cov is:\n" , mv.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    bext_max_ll, best_f1, best_f3 = -np.inf, -np.inf, -np.inf  # for question 6
    mat = np.zeros([200, 200])  # 200 X 200 matrix for the log likelihood
    for i_f1, val_f1 in enumerate(np.linspace(-10, 10, 200)):
        for i_f3, val_f3 in enumerate(np.linspace(-10, 10, 200)):
            miu = np.array([val_f1, 0, val_f3, 0])
            cur_max_ll = mv.log_likelihood(miu, cov, samples)
            mat[i_f1][i_f3] = cur_max_ll
            # for question 6
            if cur_max_ll > bext_max_ll:
                bext_max_ll = cur_max_ll
                best_f1 = val_f1
                best_f3 = val_f3

    heatmap_plot(mat, f1, f3)

    # Question 6 - Maximum likelihood
    print("best pair (f1, f3) with max loglikelihood of: " , bext_max_ll)
    print(" is: (" , best_f1, ", ", best_f3, ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
