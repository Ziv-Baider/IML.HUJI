import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_df = pd.read_csv(filename, parse_dates=[2])  # read the file
    data_df = data_df.dropna().drop_duplicates()  # rm missing  and dups

    days_of_year = []
    for day in data_df["Date"]:
        days_of_year.append(day.dayofyear)
    data_df["DayOfYear"] = days_of_year

    data_df = data_df[data_df["Temp"] > -40]
    data_df = data_df[data_df["Month"].isin(range(1, 13)) & data_df[
        "Day"].isin(range(1, 32))]

    data_df = data_df.drop("Date", 1)
    return data_df


def plot_2(df):
    """ Plots 'Daily temperature in Israel as function of day of year' - Q2 """
    temps = df["Temp"]
    daysofyear = df["DayOfYear"]
    years = df["Year"].astype("category")

    fig = px.scatter(data_frame=df, x=daysofyear, y=temps, labels={
                        "DayOfYear": "Day of year",
                        "Temp": "Temperature [farenheit]",
                        "color": "Years"}, title="Daily temperature in "
                                                 "Israel as function of day "
                                                 "of year", color=years)
    fig.show()


def bar_plot_2(df):
    """ Plots 'STD of daily temperatures as function of months' - Q2 """
    stds = df.groupby(by="Month").Temp.std()
    plt.bar(range(1, 13), stds)
    plt.xlabel("Months")
    plt.ylabel("Temperatures [farenheit]")
    plt.title("Standard deviation of daily temperatures as function of months")
    # plt.show()
    plt.savefig('Q_2 std of temp as func of month.png')


def plot_c_m(df):
    """ Plots Mean temperature as function of Month' - Q3 """
    df_gr = df.groupby(['Country', 'Month'], as_index=False).agg({"Temp": [
        'mean', 'std']})
    px.line(x=df_gr["Month"], y=df_gr[("Temp", "mean")],
            error_y=df_gr[("Temp", "std")],
            color=df_gr["Country"],
            title="Mean temperature as function of Month "
                  "with std error bar", labels={
            'x': 'Month', 'y': 'Mean Temperature'}).show()


def bar_plot_4(losses, ks):
    """ Loss of the polynomial model over the value of k' - Q4 """
    plt.bar(ks, losses)
    plt.xlabel("K")
    plt.ylabel("Loss")
    plt.title("Loss of the polynomial model over the value of k")
    # plt.show()
    plt.savefig('Q_4 Loss as func of k.png')


def fit_model_with_given_k(k, df_israel):
    """ Fits a polynomial model for a given k with Israel df. Used in Q4,Q5 """
    X, y = df_israel["DayOfYear"], df_israel["Temp"]
    p_f = PolynomialFitting(k)
    p_f.fit(X, y)
    loss = p_f._loss(X, y)
    return p_f, loss


def fit_polynomial_model_find_k(df_israel):
    """ Finds best k for min loss over polynomial model. Used in Q4,Q5 """
    losses = np.zeros(10)
    i, min_loss, min_k = 0, np.inf, 1
    ks = range(1, 11)
    for k in range(1, 11):
        _, loss = fit_model_with_given_k(k, df_israel)
        if loss < min_loss:
            min_k = k
            min_loss = loss
        losses[i] = round(loss, 2)
        print("The loss for k: " + str(k) + ", is: " + str(loss))
        i += 1
    bar_plot_4(losses, ks)
    return min_k  # minimal k


def bar_plot_5(losses, countries):
    """ Loss of polynomial model of different countries - Q5 """
    plt.bar(countries, losses)
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.title(
        "Loss of the polynomial model trained over Israel of different countries")
    # plt.show()
    plt.savefig('Q_5 Loss over different countries.png')


def fit_polynomial_model(df, min_k):
    """ Fits polynomial model trained over Israel over different countries """
    losses, countries = [], ['Israel', 'Jordan', 'The Netherlands',
                             'South '
                             'Africa']
    df_israel = df.loc[df["Country"] == "Israel"]
    df_Jordan = df.loc[df["Country"] == "Jordan"]
    df_The_Netherlands = df.loc[df["Country"] == "The Netherlands"]
    df_South_Africa = df.loc[df["Country"] == "South Africa"]

    p_f, loss = fit_model_with_given_k(min_k, df_israel)

    for df_country in [df_israel, df_Jordan, df_The_Netherlands,
                       df_South_Africa]:
        losses.append(p_f.loss(df_country.DayOfYear, df_country.Temp))

    bar_plot_5(losses, countries)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r"C:\Users\Ziv\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df.loc[df["Country"] == "Israel"]
    plot_2(df_israel)
    bar_plot_2(df_israel)

    # Question 3 - Exploring differences between countries
    plot_c_m(df)

    # Question 4 - Fitting model for different values of `k`
    min_k = fit_polynomial_model_find_k(df_israel)

    # Question 5 - Evaluating fitted model on different countries
    fit_polynomial_model(df, min_k)
