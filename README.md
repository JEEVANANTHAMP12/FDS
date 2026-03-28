# Real Estate Investment Analyzer

Single-file codebase for synthetic real-estate analytics, price modeling, investment scoring, and chart generation.

## Code Scope

- Language: Python
- Main module: `real_estate_analyzer.py`
- Style: Functional pipeline in one file with sectioned functions
- Data flow: Generate -> Clean -> Analyze -> Model -> Score -> Visualize

## Module Dependencies

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn` (`train_test_split`, `LinearRegression`, `PolynomialFeatures`, `StandardScaler`, metrics)
- `scipy.stats`
- Standard library: `argparse`, `warnings`, `sys`

## Code Architecture

`real_estate_analyzer.py` is organized into five logical sections:

1. Data generation
2. Data cleaning and exploratory analysis
3. Price prediction models
4. Investment metrics and ROI analysis
5. Visualization outputs

Orchestration is handled in `main()`.

## Core Functions

### Data

- `generate_property_dataset(n_properties=500)`
  - Creates synthetic property rows with location, type, condition, physical attributes, and costs.
  - Derives `ActualPrice` from a weighted rule plus Gaussian noise.
  - Derives `MonthlyRentEstimate` using a rent-to-price ratio.

- `clean_data(df)`
  - Drops null rows.
  - Removes price outliers using Z-score threshold (`|z| < 3`).
  - Filters invalid numeric rows (`ActualPrice`, `SquareFeet`, `Bedrooms` > 0).

### Exploratory Analysis

- `exploratory_analysis(df)`
  - Prints dataset size, price summary, and category distributions.

- `correlation_analysis(df)`
  - Builds correlation matrix for numeric features.
  - Reports sorted correlation with `ActualPrice`.

- `location_analysis(df)`
  - Aggregates location-wise statistics (mean, median, min, max, count).

### Modeling

- `train_price_models(df)`
  - One-hot encodes categorical columns.
  - Splits data train/test and standardizes features.
  - Trains:
    - Linear Regression
    - Polynomial Regression (degree 2)
  - Reports `R2`, `RMSE`, `MAE` for both models.

### Investment Analytics

- `calculate_investment_metrics(df)`
  - Adds:
    - `AnnualRentIncome`
    - `GrossYield`
    - `NetYield`
    - `CapRate`
    - `PricePerSqFt`
    - `PaybackPeriod`
    - `InvestmentScore`

- `identify_opportunities(df_metrics)`
  - Returns top 10 properties by `InvestmentScore`.

- `roi_analysis(df_metrics, investment_amount=500000, holding_period=10)`
  - Filters properties by budget.
  - Computes appreciated value at 3% annual growth.
  - Computes rental income, costs, net profit, and ROI percent.

- `investment_report(df_metrics)`
  - Reports aggregate market, yield, payback, and score distributions.

### Visualization

- `create_visualizations(df_metrics)`
  - Exports eight charts:
    1. Price distribution
    2. Correlation heatmap
    3. Yield analysis
    4. Location comparison
    5. Investment score distribution
    6. Property type analysis
    7. Price vs size
    8. Investment dashboard

## Data Schema (Generated)

Primary generated columns include:

- `PropertyID`
- `Location`
- `PropertyType`
- `YearBuilt`
- `SquareFeet`
- `Bedrooms`
- `Bathrooms`
- `Garage`
- `Condition`
- `YearsOnMarket`
- `AnnualTaxes`
- `MonthlyUtilityCost`
- `ListingPrice`
- `ActualPrice`
- `MonthlyRentEstimate`

Derived analytics columns are added later by `calculate_investment_metrics`.

## Scoring and Formula Notes

- Price synthesis (approximate rule):
  - Base + size + bed/bath + garage - age effect + location premium + condition premium + noise
- Rent estimate:
  - `MonthlyRentEstimate = ActualPrice * 0.006`
- Gross yield:
  - `AnnualRentIncome / ActualPrice * 100`
- Net yield:
  - `(AnnualRentIncome - annual_costs) / ActualPrice * 100`
- Payback period:
  - `ActualPrice / AnnualRentIncome`
- ROI projection:
  - Assumes 3% yearly appreciation and includes rental cash flow minus expenses

## Output Artifacts

- Tabular output: `property_data.csv`
- Image outputs: `01_...png` to `08_...png` from visualization section

## Code Considerations

- Polynomial regression may overfit on smaller datasets.
- Current yield logic uses a fixed rent ratio; realistic market data may require variable rent modeling.
- Investment score uses heuristic weights and can be tuned for domain-specific priorities.
