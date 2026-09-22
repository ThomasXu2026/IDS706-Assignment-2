"""Gold Price Data Analysis - Pandas Version

This module contains the Pandas analysis from the original notebook,
refactored into reusable functions so it can be tested with pytest
and used in a GitHub Actions workflow.

Dataset:
https://www.kaggle.com/datasets/mdanwarhossain200110/gold-price-2015-2025
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# ----------------------------
# Data Loading
# ----------------------------

def load_data(file_path=None):
    """
    Load the gold price CSV file.

    If no file path is provided, the function looks for
    gold_data_2015_25.csv in the same folder as this Python file.
    """

    if file_path is None:
        file_path = (
            Path(__file__).resolve().parent
            / "gold_data_2015_25.csv"
        )

    return pd.read_csv(file_path)


# ----------------------------
# Data Inspection
# ----------------------------

def inspect_data(df):
    """Print basic information about the dataset."""

    print("Shape:")
    print(df.shape)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData information:")
    df.info()

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicates:")
    print(df.duplicated().sum())


# ----------------------------
# Data Preprocessing
# ----------------------------

def preprocess_data(df):
    """
    Convert Date from string to datetime
    and create a Year column.
    """

    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year

    return df


def filter_recent_data(df, start_date="2020-01-01"):
    """
    Filter data from the given start date onward.
    Default start date is January 1, 2020.
    """

    recent_gold = df[
        df["Date"] >= start_date
    ].copy()

    return recent_gold


# ----------------------------
# Grouping and Summary
# ----------------------------

def yearly_summary(df):
    """
    Calculate yearly GLD statistics:
    mean, minimum, maximum, and count.
    """

    yearly_gold = (
        df.groupby("Year")["GLD"]
        .agg(["mean", "min", "max", "count"])
    )

    return yearly_gold


# ----------------------------
# Visualization
# ----------------------------

def plot_yearly_average(yearly_gold, show=True):
    """
    Create a line plot showing the average
    GLD value by year.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        yearly_gold.index,
        yearly_gold["mean"],
        marker="o"
    )

    ax.set_title("Average GLD Price by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average GLD Price")
    ax.set_xticks(yearly_gold.index)
    ax.grid(True)

    if show:
        plt.show()

    return fig


# ----------------------------
# 2025 Monthly Analysis
# ----------------------------

def monthly_2025_summary(df):
    """
    Calculate average monthly GLD values
    for 2025.
    """

    gold_2025 = df[
        df["Date"].dt.year == 2025
    ].copy()

    monthly_2025 = (
        gold_2025
        .groupby(
            gold_2025["Date"].dt.month
        )["GLD"]
        .mean()
        .reset_index()
    )

    monthly_2025.columns = [
        "Month",
        "Average_GLD"
    ]

    return monthly_2025


# ----------------------------
# Machine Learning
# ----------------------------

def train_linear_model(monthly_2025):
    """
    Train a Linear Regression model using:
    X = Month
    y = Average_GLD
    """

    X = monthly_2025[
        ["Month"]
    ].values

    y = monthly_2025[
        "Average_GLD"
    ]

    model = LinearRegression()

    model.fit(X, y)

    return model


def predict_september(monthly_2025):
    """
    Predict the average GLD value
    for September 2025.
    """

    model = train_linear_model(
        monthly_2025
    )

    prediction = model.predict(
        [[9]]
    )

    return prediction[0]


# ----------------------------
# Full Analysis Pipeline
# ----------------------------

def run_analysis(
    file_path=None,
    show_plot=True
):
    """
    Run the complete gold price
    analysis pipeline.
    """

    # Load data
    df = load_data(file_path)

    # Inspect original data
    inspect_data(df)

    # Preprocess data
    df = preprocess_data(df)

    # Filter data from 2020 onward
    recent_gold = filter_recent_data(
        df
    )

    print(
        "\nRows from 2020 onward:"
    )
    print(recent_gold.shape)

    # Calculate yearly statistics
    yearly_gold = yearly_summary(
        df
    )

    print(
        "\nYearly GLD summary:"
    )
    print(yearly_gold)

    # Create visualization
    plot_yearly_average(
        yearly_gold,
        show=show_plot
    )

    # Calculate monthly averages for 2025
    monthly_2025 = (
        monthly_2025_summary(df)
    )

    print(
        "\n2025 monthly average GLD:"
    )
    print(monthly_2025)

    # Predict September 2025
    predicted_sep = (
        predict_september(
            monthly_2025
        )
    )

    print(
        "\nPredicted average GLD price "
        "for September 2025: "
        f"{predicted_sep:.2f}"
    )

    return {
        "data": df,
        "recent_gold": recent_gold,
        "yearly_summary": yearly_gold,
        "monthly_2025": monthly_2025,
        "september_prediction": predicted_sep,
    }


# ----------------------------
# Run Program
# ----------------------------

if __name__ == "__main__":
    run_analysis()