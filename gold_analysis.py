"""Gold Price Data Analysis - Pandas Version

This module contains reusable functions for loading, cleaning,
summarizing, visualizing, and modeling historical gold-related data.

Dataset:
https://www.kaggle.com/datasets/mdanwarhossain200110/gold-price-2015-2025
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

DEFAULT_DATA_FILE = "gold_data_2015_25.csv"


# ----------------------------
# Data Loading
# ----------------------------


def load_data(file_path=None):
    """Load the gold price CSV file."""

    if file_path is None:
        file_path = Path(__file__).resolve().parent / DEFAULT_DATA_FILE

    return pd.read_csv(file_path)


# ----------------------------
# Data Inspection and Quality
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


def data_quality_summary(df):
    """
    Return a summary of missing values and duplicate rows.

    This makes the project data-quality checks reusable and testable.
    """

    return {
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# ----------------------------
# Data Preprocessing
# ----------------------------


def preprocess_data(df):
    """
    Convert Date from string to datetime
    and create a Year column.
    """

    processed = df.copy()

    processed["Date"] = pd.to_datetime(processed["Date"])

    processed["Year"] = processed["Date"].dt.year

    return processed


def filter_recent_data(df, start_date="2020-01-01"):
    """
    Filter data from the given start date onward.
    """

    return df[df["Date"] >= start_date].copy()


# ----------------------------
# Outlier Detection
# ----------------------------


def detect_return_outliers(df, column="GLD"):
    """
    Detect unusual daily percentage changes
    using the IQR method.

    Price levels are not used directly because
    long-term market trends can make later prices
    appear artificially unusual.

    The detected observations are retained rather
    than automatically removed because large market
    movements may represent real events.
    """

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    returns = df[column].pct_change()

    q1 = returns.quantile(0.25)
    q3 = returns.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (returns < lower_bound) | (returns > upper_bound)

    outliers = df.loc[outlier_mask].copy()

    outliers[f"{column}_Daily_Return"] = returns.loc[outlier_mask]

    return outliers


# ----------------------------
# Grouping and Summary
# ----------------------------


def yearly_summary(df):
    """
    Calculate yearly GLD statistics:
    mean, minimum, maximum, and count.
    """

    return df.groupby("Year")["GLD"].agg(
        [
            "mean",
            "min",
            "max",
            "count",
        ]
    )


def monthly_summary(df, year):
    """
    Calculate average monthly GLD values
    for a selected year.
    """

    yearly_data = df[df["Date"].dt.year == year].copy()

    if yearly_data.empty:
        raise ValueError(f"No data available for year {year}.")

    monthly_data = (
        yearly_data.groupby(yearly_data["Date"].dt.month)["GLD"].mean().reset_index()
    )

    monthly_data.columns = [
        "Month",
        "Average_GLD",
    ]

    return monthly_data


# ----------------------------
# Visualization
# ----------------------------


def plot_yearly_average(yearly_gold, show=True):
    """
    Create a line plot showing the
    average GLD value by year.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        yearly_gold.index,
        yearly_gold["mean"],
        marker="o",
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
# Machine Learning
# ----------------------------


def train_linear_model(monthly_data):
    """
    Train a Linear Regression model.

    X = Month
    y = Average_GLD
    """

    if len(monthly_data) < 2:
        raise ValueError("At least two months are " "required to train the model.")

    features = monthly_data[["Month"]].values

    target = monthly_data["Average_GLD"]

    model = LinearRegression()

    model.fit(
        features,
        target,
    )

    return model


def predict_month(monthly_data, target_month):
    """
    Predict the average GLD value
    for a selected month.
    """

    if not 1 <= target_month <= 12:
        raise ValueError("target_month must be " "between 1 and 12.")

    model = train_linear_model(monthly_data)

    prediction = model.predict([[target_month]])

    return float(prediction[0])


def backtest_month_prediction(df, year, target_month):
    """
    Backtest the linear model using a
    historical month with known data.

    The model is trained only on months
    before the target month and compared
    against the actual target-month
    average.
    """

    monthly_data = monthly_summary(
        df,
        year,
    )

    training_data = monthly_data[monthly_data["Month"] < target_month].copy()

    actual_rows = monthly_data[monthly_data["Month"] == target_month]

    if actual_rows.empty:
        raise ValueError("No actual data available " f"for {year}-{target_month:02d}.")

    prediction = predict_month(
        training_data,
        target_month,
    )

    actual = float(actual_rows["Average_GLD"].iloc[0])

    absolute_error = abs(prediction - actual)

    percentage_error = (absolute_error / actual) * 100

    return {
        "year": year,
        "target_month": target_month,
        "prediction": prediction,
        "actual": actual,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
    }


# ----------------------------
# Backward Compatibility
# ----------------------------


def monthly_2025_summary(df):
    """
    Compatibility wrapper for the
    original project function.
    """

    return monthly_summary(
        df,
        2025,
    )


def predict_september(monthly_2025):
    """
    Compatibility wrapper for the
    original project function.
    """

    return predict_month(
        monthly_2025,
        9,
    )


# ----------------------------
# Full Analysis Pipeline
# ----------------------------


def run_analysis(file_path=None, show_plot=True):
    """
    Run the complete gold price
    analysis pipeline.
    """

    df = load_data(file_path)

    inspect_data(df)

    quality = data_quality_summary(df)

    print("\nData quality summary:")
    print(quality)

    df = preprocess_data(df)

    recent_gold = filter_recent_data(df)

    print("\nRows from 2020 onward:")
    print(recent_gold.shape)

    outliers = detect_return_outliers(df)

    print("\nPotential unusual " "daily GLD movements:")
    print(len(outliers))

    yearly_gold = yearly_summary(df)

    print("\nYearly GLD summary:")
    print(yearly_gold)

    plot_yearly_average(
        yearly_gold,
        show=show_plot,
    )

    monthly_2025 = monthly_summary(
        df,
        2025,
    )

    print("\n2025 monthly " "average GLD:")
    print(monthly_2025)

    predicted_sep = predict_month(
        monthly_2025,
        9,
    )

    print(
        "\nPredicted average GLD " "price for September 2025: " f"{predicted_sep:.2f}"
    )

    backtest = backtest_month_prediction(
        df,
        year=2024,
        target_month=12,
    )

    print("\n2024 December " "backtest:")
    print("Predicted: " f"{backtest['prediction']:.2f}")
    print("Actual: " f"{backtest['actual']:.2f}")
    print("Absolute error: " f"{backtest['absolute_error']:.2f}")
    print("Percentage error: " f"{backtest['percentage_error']:.2f}%")

    return {
        "data": df,
        "data_quality": quality,
        "recent_gold": recent_gold,
        "return_outliers": outliers,
        "yearly_summary": yearly_gold,
        "monthly_2025": monthly_2025,
        "september_prediction": (predicted_sep),
        "backtest": backtest,
    }


# ----------------------------
# Run Program
# ----------------------------

if __name__ == "__main__":
    run_analysis()
