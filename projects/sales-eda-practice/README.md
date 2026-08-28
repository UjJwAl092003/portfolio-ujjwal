# Sales EDA Practice Project

An in-depth exploratory data analysis (EDA) practice project conducted on a 100-row global sales dataset using **Python, NumPy, Pandas, Matplotlib, and Seaborn**. Built as a hands-on exploration of data manipulation, summary statistics, multi-variable distributions, and business insights.

---

## Project Overview

Exploratory Data Analysis is a fundamental step in any data science and machine learning workflow. Before training predictive models, it is essential to understand feature distributions, detect anomalies, analyze correlations, and compute summary metrics.

This project methodically breaks down a multi-region sales dataset across several analytical tiers—from initial structure inspection and memory profiling to multi-level Pandas aggregations, NumPy vector operations, and Matplotlib/Seaborn statistical visualizations.

---

## Dataset Details

- **Dataset Size:** 100 orders &times; 14 attributes
- **Missing Values:** 0 null values across all columns
- **Duplicate Records:** 0 duplicates
- **Memory Footprint:** ~16.9 KB

### Column Schema

| Column | Type | Category | Description |
|---|---|---|---|
| `Region` | String | Categorical | Geographic sales region (7 unique regions) |
| `Country` | String | Categorical | Destination country (76 unique nations) |
| `Item Type` | String | Categorical | Product category (12 unique item types) |
| `Sales Channel` | String | Categorical | Purchase channel: `Online` vs `Offline` (50/50 split) |
| `Order Priority` | String | Categorical | Priority code: `H` (High), `L` (Low), `C` (Critical), `M` (Medium) |
| `Order Date` | Datetime | Temporal | Date order was placed (100 distinct dates) |
| `Order ID` | Integer | Identifier | Unique transaction identifier |
| `Ship Date` | Datetime | Temporal | Date order was fulfilled (99 distinct dates) |
| `Units Sold` | Integer | Numerical | Number of product units in transaction |
| `Unit Price` | Float | Numerical | Price per unit ($) |
| `Unit Cost` | Float | Numerical | Production / wholesale cost per unit ($) |
| `Total Revenue` | Float | Numerical | Gross transaction revenue ($) |
| `Total Cost` | Float | Numerical | Total wholesale cost ($) |
| `Total Profit` | Float | Numerical | Net transaction profit ($) |

---

## Analysis & Methodology

### 1. Data Understanding & Structural Inspection

The initial phase focuses on shape validation, data types, missing value audit, and duplicate detection:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
sales = pd.read_csv('data/sales.csv')

# Dataset shape and structure
print("Shape:", sales.shape) # (100, 14)
print("Missing values per column:\n", sales.isnull().sum())
print("Duplicate rows:", sales.duplicated().sum())

# Categorical vs Numerical Column Split
nums_cols = sales.select_dtypes(include='number').columns
cat_cols = sales.select_dtypes(include='object').columns
```

### 2. Summary Statistics & Aggregations

Evaluating central tendencies, order volumes, and financial metrics across dimensions:

```python
# Summary statistics across numerical features
sales[nums_cols].describe().T

# Orders by Region
region_orders = sales.groupby('Region')['Order ID'].count().sort_values(ascending=False)

# Orders by Item Type
item_orders = sales.groupby('Item Type')['Order ID'].count().sort_values(ascending=False)

# Total Revenue by Region
region_revenue = sales.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False)
```

**Key Findings:**
- **Sub-Saharan Africa** leads in both order count (**36 orders**) and total gross revenue (**$39.67M**).
- **Europe** is second in revenue (**$33.37M**) across 22 orders.
- **Clothes** and **Cosmetics** are the most frequently ordered item types (**13 orders each**).
- **Sales Channel** is an exact 50/50 split (**50 Online, 50 Offline**).

### 3. NumPy Array Operations & Statistical Calculations

Direct array manipulation practicing vectorized statistics, percentiles, normalization, and coordinate search:

```python
# Extract revenue as NumPy ndarray
revenue = sales['Total Revenue'].to_numpy()

# Core Statistics
mean_rev = np.mean(revenue)
median_rev = np.median(revenue)
std_rev = np.std(revenue)

# Quartiles and Percentiles
p25, p50, p75 = np.percentile(revenue, [25, 50, 75])

# Values relative to mean
above_mean_count = np.sum(revenue > mean_rev)
below_mean_count = np.sum(revenue < mean_rev)

# Z-Score Normalization (Standardization)
revenue_normalized = (revenue - mean_rev) / std_rev

# Extremes by index
max_idx = np.argmax(revenue)
min_idx = np.argmin(revenue)
```

### 4. Data Visualizations

Using Matplotlib and Seaborn to visualize patterns across categorical and continuous variables:

- **Regional Breakdown:** Bar charts mapping order counts and total revenue per region.
- **Sales Channels:** Pie charts and count plots showing the channel distribution.
- **Time Series:** Daily order volume mapped over time using parsed datetime indices.
- **Scatter Plot (Units Sold vs Profit):** Examining relationship between volume sold and net profit generated.
- **Correlation Heatmap:** Computing pairwise Pearson correlation coefficients across numerical attributes.

```python
# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = sales[['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
```

---

## Key Takeaways

1. **Volume Strongly Correlates with Total Profit:** Units Sold has a strong positive correlation with Total Profit and Total Revenue.
2. **Channel Parity:** Online and Offline channels generate equal transaction frequency (50/50), showing balanced operational adoption.
3. **Regional Revenue Leadership:** Sub-Saharan Africa and Europe represent over 53% of total revenue in this dataset.
4. **Data Hygiene:** The dataset has zero missing values and zero duplicate rows, enabling direct exploration without imputation artifacts.

---

## Technologies Used

- **Python 3**
- **NumPy** — Vector operations, percentiles, z-score normalization
- **Pandas** — Data cleaning, multi-index aggregation, grouping
- **Matplotlib & Seaborn** — Histograms, scatter plots, correlation heatmaps, bar charts
- **Jupyter Notebook** — Interactive data exploration environment

---
