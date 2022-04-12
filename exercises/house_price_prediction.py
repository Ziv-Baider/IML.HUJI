import matplotlib.pyplot as plt

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.metrics import mean_square_error
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data_df = pd.read_csv(filename)  # read the file
    data_df = data_df.dropna().drop_duplicates()  # rm missing  and dups
    data_df = data_df.drop(["id", "date", "lat", "long"], axis=1)

    features_greater_than_0 = ["price", "sqft_living", "sqft_lot",
                               "sqft_above", "yr_built", "sqft_living15",
                               "sqft_lot15"]
    features_greater_or_equal_0 = ["bathrooms", "floors", "sqft_basement",
                                   "yr_renovated"]
    for column in features_greater_than_0:
        data_df = data_df[data_df[column] > 0]

    for column in features_greater_or_equal_0:
        data_df = data_df[data_df[column] >= 0]

    # assuring that the ranged values are in the correct range acc. to docum.
    data_df = data_df[data_df["view"].isin(range(0, 5)) & data_df[
        "condition"].isin(range(1, 6)) & data_df["grade"].isin(range(1, 14))]

    # remove rows that don't make sense
    data_df = data_df[data_df["bedrooms"] < 15]
    data_df = data_df[data_df["sqft_lot"] < 1000000]
    data_df = data_df[data_df["sqft_lot15"] < 1000000]

    # turn categorical variable yr_renovated into 4 different grades
    max_yr_renovated = data_df["yr_renovated"].max()
    cond_a = data_df["yr_renovated"] > (max_yr_renovated - 15)
    cond_b = ((max_yr_renovated - 40) < data_df["yr_renovated"]) & (data_df[
                                                "yr_renovated"] < (
                                                    max_yr_renovated - 15))
    cond_c = (0 < data_df["yr_renovated"]) & (data_df["yr_renovated"] < (
            max_yr_renovated - 40))

    data_df["recently_renovated"] = np.where(cond_a, 3, 0)
    data_df["not_recently_renovated"] = np.where(cond_b, 2, 0)
    data_df["once_renovated"] = np.where(cond_c, 1, 0)
    data_df = data_df.drop("yr_renovated", 1)  # remove the categorical from df

    # turn categorical variable yr_built into 3 different grades
    max_yr_built = data_df["yr_built"].max()
    cond_d = data_df["yr_built"] > (max_yr_built - 30)
    cond_e = ((max_yr_built - 60) < data_df["yr_built"]) & (data_df[
                                            "yr_built"] < (
                                                            max_yr_built - 30))
    cond_f = (0 < data_df["yr_built"]) & (data_df["yr_built"] < (max_yr_built
                                                                 - 60))
    data_df["recently_built"] = np.where(cond_d, 3, 0)
    data_df["not_recently_built"] = np.where(cond_e, 2, 0)
    data_df["a_long_time_ago_built"] = np.where(cond_f, 1, 0)

    data_df["yr_built_normalized"] = (data_df["yr_built"] / 10).astype(int)
    data_df = data_df.drop("yr_built", 1)  # remove the categorical from df

    # change categorical variable zipcode to dummy variable
    data_df = pd.get_dummies(data_df, prefix='zipcode_', columns=['zipcode'])

    # return Design matrix and response vector as a tuple
    return data_df.drop("price", 1), data_df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # remove all the dummy zipcodes I crated earlier
    X = X.loc[:, ~(X.columns.str.contains('zipcode_', case=False))]

    for feature in X:
        cov_feature_response = np.cov(X[feature], y)[0, 1]
        std_feat_res_mul = np.std(X[feature]) * np.std(y)
        p_cor = cov_feature_response / std_feat_res_mul
        fig = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x",
                         y="y", title="Pearson Correlation between " +
                                      feature +
                                      " and Response " + str(p_cor),
                         labels={"x": "feature: " + feature, "y": "response"})
        fig.write_image("Pearson_Corr elation.%s.png" % feature)
        # fig.show()


def plot_mean_loss_af_p(x, loss_mean, loss_std):
    """ plots mean loss as function of precent of train set with confidence
    interval """
    fig = go.Figure([go.Scatter(x=x,
                                y=loss_mean,
                                mode='lines'),
                     go.Scatter(x=x,
                                y=loss_mean + 2 * loss_std,
                                mode='lines', marker=dict(color='#444'),
                                showlegend=False),
                     go.Scatter(x=x,
                                y=loss_mean - 2 * loss_std,
                                mode='lines', marker=dict(color='#444'),
                                showlegend=False,
                                fillcolor='rgba(68,68,68,0.3)',
                                fill='tonexty')])

    fig.update_xaxes(ticksuffix="%", title_text="Percentage of "
                                                "train dataset [%]")
    fig.update_yaxes(title_text="Mean Loss of test dataset")
    fig.update_layout(
        title_text="Mean loss as function of train dataset, and error ribbon")
    fig.write_image("loss.png")
    # fig.show()


def fit_linear_regression_increasing_model(train_X, train_y, test_X, test_y):
    """ Fits linear regression model ober increasing precent of train set """
    mean_loss, std_loss, ps = np.zeros(91), np.zeros(91), np.zeros(91)
    means, stds = np.zeros(100), np.zeros(100)
    for p in range(10, 101):
        l_r = LinearRegression()

        precent = p / 100
        losses = np.zeros(10)
        for i in range(10):
            train_X_sample, train_y_sample, test_X_samle, test_y_samle = \
                split_train_test(train_X, train_y, precent)
            l_r.fit(train_X_sample.to_numpy(), train_y_sample)
            loss = l_r.loss(test_X, test_y)  # calculate loss
            losses[i] = loss

        means[p-10] = np.mean(losses)
        stds[p-10] = np.std(losses)
        ps[p-10] = p

    plot_mean_loss_af_p(ps, means, stds)

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(r"C:\Users\Ziv\datasets\house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, .75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    fit_linear_regression_increasing_model(train_X, train_y, test_X, test_y)

    # y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    # y_pred = np.array(
    #     [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275,
    #      563867.1347574, 395102.94362135])
    # print(mean_square_error(y_true,y_pred))
