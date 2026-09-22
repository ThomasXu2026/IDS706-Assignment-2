import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from gold_analysis import (
    load_data,
    preprocess_data,
    filter_recent_data,
    yearly_summary,
    plot_yearly_average,
    monthly_2025_summary,
    predict_september,
    run_analysis,
)


def test_load_data():
    """The project CSV should load with the expected shape and columns."""
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2666, 6)
    assert list(df.columns) == [
        "Date",
        "SPX",
        "GLD",
        "USO",
        "SLV",
        "EUR/USD",
    ]


def test_missing_file(tmp_path):
    """Loading a file that does not exist should raise FileNotFoundError."""
    missing_file = tmp_path / "missing_gold_data.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_file)


def test_preprocess_data():
    """Date should be converted to datetime and Year should be created."""
    df = load_data()
    processed = preprocess_data(df)

    assert pd.api.types.is_datetime64_any_dtype(processed["Date"])
    assert "Year" in processed.columns
    assert processed["Year"].min() == 2015
    assert processed["Year"].max() == 2025

    # The preprocessing function should not modify the original DataFrame.
    assert "Year" not in df.columns


def test_filter_recent_data():
    """Filtering from 2020 onward should keep only dates from 2020 or later."""
    df = preprocess_data(load_data())
    recent = filter_recent_data(df)

    assert len(recent) == 1411
    assert recent["Date"].min() >= pd.Timestamp("2020-01-01")
    assert recent["Year"].min() == 2020


def test_yearly_summary():
    """Yearly summary should contain the expected years and statistics."""
    df = preprocess_data(load_data())
    summary = yearly_summary(df)

    assert len(summary) == 11
    assert list(summary.columns) == ["mean", "min", "max", "count"]
    assert summary.loc[2020, "count"] == 253
    assert summary.loc[2020, "mean"] == pytest.approx(
        166.653755,
        abs=0.000001,
    )


def test_monthly_2025_summary():
    """The 2025 monthly summary should contain January through August."""
    df = preprocess_data(load_data())
    monthly = monthly_2025_summary(df)

    assert len(monthly) == 8
    assert monthly["Month"].tolist() == list(range(1, 9))
    assert monthly.loc[
        monthly["Month"] == 1,
        "Average_GLD",
    ].iloc[0] == pytest.approx(
        250.4255,
        abs=0.0001,
    )


def test_yearly_plot():
    """The visualization function should create the expected line chart."""
    df = preprocess_data(load_data())
    summary = yearly_summary(df)

    fig = plot_yearly_average(summary, show=False)
    ax = fig.axes[0]

    assert fig is not None
    assert ax.get_title() == "Average GLD Price by Year"
    assert ax.get_xlabel() == "Year"
    assert ax.get_ylabel() == "Average GLD Price"
    assert len(ax.lines) == 1

    plt.close(fig)


def test_september_prediction():
    """The Linear Regression model should reproduce the September prediction."""
    df = preprocess_data(load_data())
    monthly = monthly_2025_summary(df)

    prediction = predict_september(monthly)

    assert prediction > 0
    assert prediction == pytest.approx(
        328.83,
        abs=0.01,
    )


def test_full_pipeline():
    """
    System/integration test:
    the complete workflow should run successfully from CSV loading
    through preprocessing, analysis, visualization, and prediction.
    """
    results = run_analysis(show_plot=False)

    assert results["data"].shape == (2666, 7)
    assert len(results["recent_gold"]) == 1411
    assert len(results["yearly_summary"]) == 11
    assert len(results["monthly_2025"]) == 8
    assert results["september_prediction"] == pytest.approx(
        328.83,
        abs=0.01,
    )
