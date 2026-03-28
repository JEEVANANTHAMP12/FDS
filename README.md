# 🏠 Real Estate Investment Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0.0-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> A comprehensive, single-file Python application for synthetic real estate analytics — covering price modelling, investment scoring, ROI analysis, and chart generation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Core Functions](#core-functions)
- [Data Schema](#data-schema)
- [Scoring & Formulas](#scoring--formulas)
- [Output Artifacts](#output-artifacts)
- [Notes & Limitations](#notes--limitations)

---

## Overview

The **Real Estate Investment Analyzer** generates a synthetic property dataset, cleans it, performs exploratory data analysis, trains price prediction models, computes investment metrics, and produces publication-ready visualizations — all within a single Python file.

**Data flow:**

```
Generate → Clean → Analyze → Model → Score → Visualize
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Synthetic Dataset** | Generates 500+ realistic property records with noise |
| 🧹 **Data Cleaning** | Z-score outlier removal and validation filters |
| 🔍 **Exploratory Analysis** | Price summaries, correlations, and location breakdowns |
| 🤖 **ML Price Models** | Linear Regression & Polynomial Regression (degree 2) |
| 💰 **Investment Scoring** | Yield, Cap Rate, Payback Period, and composite score |
| 📈 **ROI Projection** | 10-year holding-period analysis at 3% annual growth |
| 🖼️ **8 Charts** | Distribution, heatmap, yield, dashboard, and more |
| 📁 **CSV Export** | Full dataset with all derived columns saved to disk |

---

## 🗂️ Project Structure

```
FDS/
├── real_estate_analyzer.py   # Main application (all logic in one file)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.8+ |
| pandas | 2.0.0 |
| numpy | 1.24.0 |
| scikit-learn | 1.2.0 |
| matplotlib | 3.7.0 |
| seaborn | 0.12.0 |
| scipy | 1.10.0 |

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/JEEVANANTHAMP12/FDS.git
   cd FDS
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS / Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

Run the analyzer with default settings (500 properties):

```bash
python real_estate_analyzer.py
```

The script will:
1. Generate a synthetic property dataset
2. Clean and validate the data
3. Print exploratory analysis to the console
4. Train and evaluate price prediction models
5. Compute investment metrics and identify top opportunities
6. Export `property_data.csv` and eight PNG chart files

---

## 🔄 Data Pipeline

```
generate_property_dataset()
        │
        ▼
    clean_data()
        │
        ├──► exploratory_analysis()
        ├──► correlation_analysis()
        ├──► location_analysis()
        │
        ▼
 train_price_models()
        │
        ▼
calculate_investment_metrics()
        │
        ├──► identify_opportunities()
        ├──► roi_analysis()
        └──► investment_report()
                │
                ▼
      create_visualizations()
```

`main()` orchestrates all steps in the above order.

---

## 🧩 Core Functions

### Data Generation

#### `generate_property_dataset(n_properties=500)`
- Creates synthetic property rows with location, type, condition, physical attributes, and costs.
- Derives `ActualPrice` from a weighted rule plus Gaussian noise.
- Derives `MonthlyRentEstimate` using a rent-to-price ratio (`× 0.006`).

#### `clean_data(df)`
- Drops null rows.
- Removes price outliers using Z-score threshold (`|z| < 3`).
- Filters invalid numeric rows (`ActualPrice`, `SquareFeet`, `Bedrooms` > 0).

---

### Exploratory Analysis

#### `exploratory_analysis(df)`
- Prints dataset size, total/average/median price, and category distributions.

#### `correlation_analysis(df)`
- Builds a numeric correlation matrix.
- Reports features sorted by correlation with `ActualPrice`.

#### `location_analysis(df)`
- Aggregates location-wise statistics: mean, median, min, max, count.

---

### Modelling

#### `train_price_models(df)`
- One-hot encodes categorical columns.
- Splits data 80/20 train/test and standardizes features.
- Trains two models:
  - **Linear Regression**
  - **Polynomial Regression** (degree 2)
- Reports `R²`, `RMSE`, and `MAE` for both models.

---

### Investment Analytics

#### `calculate_investment_metrics(df)`
Adds the following derived columns:

| Column | Description |
|---|---|
| `AnnualRentIncome` | `MonthlyRentEstimate × 12` |
| `GrossYield` | `AnnualRentIncome / ActualPrice × 100` |
| `NetYield` | `(AnnualRentIncome − annual_costs) / ActualPrice × 100` |
| `CapRate` | Net operating income / property value |
| `PricePerSqFt` | `ActualPrice / SquareFeet` |
| `PaybackPeriod` | `ActualPrice / AnnualRentIncome` |
| `InvestmentScore` | Weighted composite score |

#### `identify_opportunities(df_metrics)`
- Returns the **top 10 properties** ranked by `InvestmentScore`.

#### `roi_analysis(df_metrics, investment_amount=500000, holding_period=10)`
- Filters properties within budget.
- Computes appreciated value at **3% annual growth**.
- Reports rental income, costs, net profit, and total ROI %.

#### `investment_report(df_metrics)`
- Prints aggregate market, yield, payback, and score distributions.

---

### Visualization

#### `create_visualizations(df_metrics)`
Exports **eight PNG charts**:

| # | Chart |
|---|---|
| 1 | Price distribution |
| 2 | Correlation heatmap |
| 3 | Yield analysis |
| 4 | Location comparison |
| 5 | Investment score distribution |
| 6 | Property type analysis |
| 7 | Price vs. square footage |
| 8 | Investment dashboard |

---

## 🗃️ Data Schema

### Primary Columns (Generated)

| Column | Type | Description |
|---|---|---|
| `PropertyID` | int | Unique property identifier |
| `Location` | str | Downtown, Suburbs, Waterfront, Mountains, Business District, Residential Area |
| `PropertyType` | str | Apartment, House, Condo, Townhouse, Commercial |
| `YearBuilt` | int | Year of construction (1980–2023) |
| `SquareFeet` | int | Interior area (800–5000 sq ft) |
| `Bedrooms` | int | Number of bedrooms (1–5) |
| `Bathrooms` | int | Number of bathrooms (1–3) |
| `Garage` | int | Garage spaces (0–2) |
| `Condition` | str | Excellent, Good, Fair, Needs Repair |
| `YearsOnMarket` | int | Days listed (0–20) |
| `AnnualTaxes` | int | Annual property taxes ($2,000–$15,000) |
| `MonthlyUtilityCost` | int | Monthly utility estimate ($100–$500) |
| `ListingPrice` | int | Listed price ($150,000–$800,000) |
| `ActualPrice` | float | Computed fair value |
| `MonthlyRentEstimate` | int | Estimated monthly rent |

### Derived Analytics Columns

Added by `calculate_investment_metrics()`:
`AnnualRentIncome`, `GrossYield`, `NetYield`, `CapRate`, `PricePerSqFt`, `PaybackPeriod`, `InvestmentScore`

---

## 📐 Scoring & Formulas

### Price Synthesis
```
ActualPrice ≈ 50,000
            + SquareFeet × 150
            + Bedrooms × 30,000
            + Bathrooms × 20,000
            + Garage × 15,000
            − (2024 − YearBuilt) × 500
            + location_premium   (Waterfront: +100k, Downtown: +50k)
            + condition_premium  (Excellent: +50k)
            + N(0, 20,000)
```

### Key Formulas
| Metric | Formula |
|---|---|
| Monthly Rent | `ActualPrice × 0.006` |
| Gross Yield | `AnnualRentIncome / ActualPrice × 100` |
| Net Yield | `(AnnualRentIncome − annual_costs) / ActualPrice × 100` |
| Payback Period | `ActualPrice / AnnualRentIncome` |
| Appreciated Value | `ActualPrice × (1.03) ^ holding_period` |

---

## 📦 Output Artifacts

| File | Description |
|---|---|
| `property_data.csv` | Full dataset with all generated and derived columns |
| `01_price_distribution.png` | Histogram of property prices |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_yield_analysis.png` | Gross vs. net yield comparison |
| `04_location_comparison.png` | Average metrics by location |
| `05_investment_score.png` | Score distribution |
| `06_property_type_analysis.png` | Type-based breakdown |
| `07_price_vs_size.png` | Scatter plot of price vs. sq ft |
| `08_investment_dashboard.png` | Combined multi-panel dashboard |

---

## ⚠️ Notes & Limitations

- **Polynomial regression** may overfit on smaller datasets; use with caution when reducing `n_properties`.
- **Rent estimates** use a fixed ratio (`ActualPrice × 0.006` per month, i.e. 0.6% of property value per month); real-world modeling may require variable rent data.
- **Investment score** relies on heuristic weights which can be tuned for domain-specific priorities.
- All data is **synthetic** and intended for educational/demonstration purposes only.
