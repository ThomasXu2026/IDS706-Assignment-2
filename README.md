# Gold Price Data Analysis: Pandas and Polars

## Project Overview

This project analyzes historical gold-related market data from 2015 to 2025.  
The same analysis is completed twice: first with **Pandas** and then with **Polars**.

The project includes:

- Importing and inspecting the dataset
- Checking missing values and duplicates
- Filtering recent data
- Grouping gold data by year
- Visualizing yearly average GLD values
- Calculating monthly average GLD values for 2025
- Using Linear Regression to predict the average GLD value for September 2025
- Comparing the Pandas and Polars results

## Dataset

Source: Kaggle  
Dataset: Gold Price 2015-2025  
https://www.kaggle.com/datasets/mdanwarhossain200110/gold-price-2015-2025

The dataset contains **2,666 rows and 6 columns**:

- `Date`
- `SPX`
- `GLD`
- `USO`
- `SLV`
- `EUR/USD`

The data runs from January 2015 through August 14, 2025, so the 2025 data is not a complete year.

## Libraries Used

- pandas
- polars
- matplotlib
- scikit-learn

## Setup

1. Download the dataset file `gold_data_2015_25.csv` from Kaggle

2. Place `gold_data_2015_25.csv` in the same folder as `w2_mini_da.ipynb`.

3. Install the required Python libraries:
   pip install pandas polars matplotlib scikit-learn
4. Open the notebook file with the coding tool of your choice, and run all cells from top to bottom.

## Pandas Analysis

### 1. Import and Inspect the Data

The dataset was loaded using `pd.read_csv()`.

I used to inspect the data:

- `df.shape`
- `df.head()`
- `df.info()`
- `df.describe()`
- `df.isnull().sum()`
- `df.duplicated().sum()`

The dataset contains:

- **2,666 rows**
- **6 original columns**
- **0 missing values**
- **0 duplicate rows**

### 2. Filtering and Grouping

The `Date` column was converted to datetime format.

I filtered the dataset to include data from **2020 onward**, which produced **1,411 rows**.

A new `Year` column was created from the `Date` column.  
The data was then grouped by year to calculate:

- Mean GLD value
- Minimum GLD value
- Maximum GLD value
- Number of observations

Some yearly average GLD values were:

- 2015: about **111.15**
- 2020: about **166.65**
- 2023: about **180.45**
- 2024: about **221.10**
- 2025: about **288.87**

The 2025 average is based only on the available data through August 14, 2025.

### 3. Visualization

A line plot was created to show the **average GLD value by year from 2015 to 2025**.

The plot shows that the yearly average GLD value generally increased over time, with stronger growth in the later years. Since the 2025 date ended with August, therefore the 2025 data show in the plot has only represent the avergae yearly gold price using date from January to August.

### 4. Machine Learning

For the machine learning experiment, I filtered the data to **2025** and calculated the average GLD value for each month from January through August.

The model used:

- **Input (X):** Month
- **Output (y):** Average monthly GLD value
- **Algorithm:** Linear Regression from scikit-learn

The model was trained on months 1 through 8 and then used to predict month 9.

The Pandas-based analysis predicted:

**September 2025 average GLD = 328.83**

---

## Polars Analysis

The same data analysis steps were repeated using **Polars**.

### 1. Import and Inspect the Data

The dataset was loaded with `pl.read_csv()`.

The data is inspected using:

- `df_pl.shape`
- `df_pl.head()`
- `df_pl.schema`
- `df_pl.describe()`
- `df_pl.null_count()`
- `df_pl.is_duplicated().sum()`

Polars also found:

- **2,666 rows**
- **6 original columns**
- **0 null values**
- **0 duplicate rows**

### 2. Filtering and Grouping

The `Date` column was converted from string format to the Polars `Date` type.

The data was filtered to records from **2020 onward**, producing the same **1,411 rows** as the Pandas analysis.

A `Year` column was created and the data was grouped by year using `group_by()`.

The same yearly GLD statistics were calculated:

- Mean
- Minimum
- Maximum
- Count

### 3. Visualization

The Polars yearly summary was used with Matplotlib to create the same yearly average GLD line plot.

### 4. Machine Learning

The 2025 data was filtered using Polars and grouped by month.

The monthly average GLD values for January through August were converted to NumPy arrays with `.to_numpy()` so they could be used by scikit-learn.

The same Linear Regression model was then used.

The Polars-based analysis predicted:

**September 2025 average GLD = 328.83**

---

## Pandas vs. Polars

Both Pandas and Polars produced the same main analysis results.
Pandas uses methods such as `groupby()`, while Polars uses expressions and methods such as `group_by()`, `filter()`, and `with_columns()`.

## Main Findings

- The dataset is clean, with no missing values or duplicate rows.
- Average GLD values generally increased from 2015 to 2025.
- The increase became much stronger in the later years of the dataset.
- Pandas and Polars produced matching results for the filtering, grouping, and machine learning analysis.
- The Linear Regression model predicted an average GLD value of approximately **328.83** for September 2025.


## Files

- `w2_mini_da.ipynb` - Jupyter Notebook containing the Pandas and Polars analysis
- `README.md` - Project documentation
