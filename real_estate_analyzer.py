"""
REAL ESTATE INVESTMENT ANALYZER - Single File Version
Comprehensive system for analyzing real estate investment opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

# ======================================================================
# SECTION 1: DATA GENERATION
# ======================================================================

def generate_property_dataset(n_properties=500):
    """Generate a realistic property dataset for Real Estate Investment Analysis"""
    np.random.seed(42)
    
    locations = ['Downtown', 'Suburbs', 'Waterfront', 'Mountains', 'Business District', 'Residential Area']
    property_types = ['Apartment', 'House', 'Condo', 'Townhouse', 'Commercial']
    condition_ratings = ['Excellent', 'Good', 'Fair', 'Needs Repair']
    
    data = {
        'PropertyID': range(1001, 1001 + n_properties),
        'Location': np.random.choice(locations, n_properties),
        'PropertyType': np.random.choice(property_types, n_properties),
        'YearBuilt': np.random.randint(1980, 2024, n_properties),
        'SquareFeet': np.random.randint(800, 5000, n_properties),
        'Bedrooms': np.random.randint(1, 6, n_properties),
        'Bathrooms': np.random.randint(1, 4, n_properties),
        'Garage': np.random.randint(0, 3, n_properties),
        'Condition': np.random.choice(condition_ratings, n_properties),
        'YearsOnMarket': np.random.randint(0, 20, n_properties),
        'AnnualTaxes': np.random.randint(2000, 15000, n_properties),
        'MonthlyUtilityCost': np.random.randint(100, 500, n_properties),
        'ListingPrice': np.random.randint(150000, 800000, n_properties),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate actual price based on features
    df['ActualPrice'] = (
        50000 +
        df['SquareFeet'] * 150 +
        df['Bedrooms'] * 30000 +
        df['Bathrooms'] * 20000 +
        df['Garage'] * 15000 +
        (2024 - df['YearBuilt']) * (-500) +
        np.where(df['Location'] == 'Waterfront', 100000, 0) +
        np.where(df['Location'] == 'Downtown', 50000, 0) +
        np.where(df['Condition'] == 'Excellent', 50000, 0) +
        np.random.normal(0, 20000, n_properties)
    )
    
    df['ActualPrice'] = df['ActualPrice'].abs()
    df['MonthlyRentEstimate'] = (df['ActualPrice'] * 0.006).astype(int)
    
    return df.sort_values('ActualPrice', ascending=False).reset_index(drop=True)

# ======================================================================
# SECTION 2: DATA CLEANING & ANALYSIS
# ======================================================================

def clean_data(df):
    """Clean and validate the dataset"""
    print("\n" + "="*60)
    print("DATA CLEANING AND VALIDATION")
    print("="*60)
    
    original_count = len(df)
    df = df.copy()
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    df = df.dropna()
    print(f"\n[OK] Removed {original_count - len(df)} rows with missing values")
    
    z_scores = np.abs(stats.zscore(df['ActualPrice']))
    outliers_before = len(df)
    df = df[z_scores < 3]
    print(f"[OK] Removed {outliers_before - len(df)} price outliers")
    
    df = df[df['ActualPrice'] > 0]
    df = df[df['SquareFeet'] > 0]
    df = df[df['Bedrooms'] > 0]
    print(f"[OK] Cleaned dataset: {len(df)} valid properties")
    
    return df

def exploratory_analysis(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"Total Properties: {len(df)}")
    print(f"Total Value: ${df['ActualPrice'].sum():,.0f}")
    print(f"Average Price: ${df['ActualPrice'].mean():,.0f}")
    print(f"Median Price: ${df['ActualPrice'].median():,.0f}")
    print(f"Price Range: ${df['ActualPrice'].min():,.0f} - ${df['ActualPrice'].max():,.0f}")
    
    print(f"\nProperty Types Distribution:")
    print(df['PropertyType'].value_counts())
    
    print(f"\nLocation Distribution:")
    print(df['Location'].value_counts())
    
    print(f"\nCondition Distribution:")
    print(df['Condition'].value_counts())
    
    print(f"\nBasic Statistics:")
    print(df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'ActualPrice']].describe())

def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    numeric_cols = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Garage', 
                   'YearBuilt', 'AnnualTaxes', 'MonthlyUtilityCost', 
                   'ActualPrice', 'MonthlyRentEstimate']
    
    correlation_matrix = df[numeric_cols].corr()
    price_correlations = correlation_matrix['ActualPrice'].sort_values(ascending=False)
    
    print("\nCorrelation with Property Price:")
    print(price_correlations)
    
    return correlation_matrix

def location_analysis(df):
    """Analyze properties by location"""
    print("\n" + "="*60)
    print("LOCATION-BASED ANALYSIS")
    print("="*60)
    
    location_stats = df.groupby('Location').agg({
        'ActualPrice': ['mean', 'median', 'min', 'max', 'count'],
        'SquareFeet': 'mean',
        'Bedrooms': 'mean',
        'MonthlyRentEstimate': 'mean'
    }).round(0)
    
    print("\nPrice Statistics by Location:")
    print(location_stats)
    
    return location_stats

# ======================================================================
# SECTION 3: PRICE PREDICTION
# ======================================================================

def train_price_models(df):
    """Train linear and polynomial regression models"""
    print("\n" + "="*60)
    print("PRICE PREDICTION MODELS")
    print("="*60)
    
    # Prepare features
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['Location', 'PropertyType', 'Condition'], drop_first=True)
    
    feature_cols = [col for col in df_encoded.columns 
                   if col not in ['PropertyID', 'ActualPrice', 'ListingPrice', 'MonthlyRentEstimate']]
    
    X = df_encoded[feature_cols]
    y = df_encoded['ActualPrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Linear Model
    print("\n[LINEAR REGRESSION MODEL]")
    model_linear = LinearRegression()
    model_linear.fit(X_train_scaled, y_train)
    
    y_pred_train = model_linear.predict(X_train_scaled)
    y_pred_test = model_linear.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training R2 Score: {train_r2:.4f}")
    print(f"Testing R2 Score: {test_r2:.4f}")
    print(f"Training RMSE: ${train_rmse:,.0f}")
    print(f"Testing RMSE: ${test_rmse:,.0f}")
    print(f"Mean Absolute Error: ${test_mae:,.0f}")
    
    # Polynomial Model
    print("\n[POLYNOMIAL REGRESSION MODEL (Degree=2)]")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    
    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train)
    
    y_pred_train = model_poly.predict(X_train_poly)
    y_pred_test = model_poly.predict(X_test_poly)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training R2 Score: {train_r2:.4f}")
    print(f"Testing R2 Score: {test_r2:.4f}")
    print(f"Training RMSE: ${train_rmse:,.0f}")
    print(f"Testing RMSE: ${test_rmse:,.0f}")
    print(f"Mean Absolute Error: ${test_mae:,.0f}")
    
    return model_linear, scaler

# ======================================================================
# SECTION 4: INVESTMENT ANALYSIS
# ======================================================================

def calculate_investment_metrics(df):
    """Calculate key investment metrics for each property"""
    print("\n" + "="*60)
    print("INVESTMENT METRICS CALCULATION")
    print("="*60)
    
    df_analysis = df.copy()
    
    df_analysis['AnnualRentIncome'] = df_analysis['MonthlyRentEstimate'] * 12
    df_analysis['GrossYield'] = (df_analysis['AnnualRentIncome'] / df_analysis['ActualPrice'] * 100).round(2)
    
    annual_costs = df_analysis['AnnualTaxes'] + (df_analysis['MonthlyUtilityCost'] * 12)
    df_analysis['NetYield'] = ((df_analysis['AnnualRentIncome'] - annual_costs) / df_analysis['ActualPrice'] * 100).round(2)
    
    df_analysis['CapRate'] = df_analysis['GrossYield']
    df_analysis['PricePerSqFt'] = (df_analysis['ActualPrice'] / df_analysis['SquareFeet']).round(0)
    df_analysis['PaybackPeriod'] = (df_analysis['ActualPrice'] / (df_analysis['AnnualRentIncome'] + 1)).round(2)
    
    df_analysis['InvestmentScore'] = (
        (df_analysis['GrossYield'] * 10) +
        (100 - df_analysis['PaybackPeriod']) +
        (df_analysis['Bedrooms'] * 5) +
        (1 / (df_analysis['PricePerSqFt'] / 1000) * 20)
    ).round(2)
    
    return df_analysis

def identify_opportunities(df_metrics):
    """Identify top investment opportunities"""
    print("\n" + "="*60)
    print("TOP 10 INVESTMENT OPPORTUNITIES")
    print("="*60)
    
    top_properties = df_metrics.nlargest(10, 'InvestmentScore')[
        ['PropertyID', 'Location', 'PropertyType', 'ActualPrice', 'GrossYield', 
         'NetYield', 'PaybackPeriod', 'InvestmentScore']
    ].reset_index(drop=True)
    
    print("\n" + top_properties.to_string(index=False))
    
    return top_properties

def roi_analysis(df_metrics, investment_amount=500000, holding_period=10):
    """Analyze Return on Investment"""
    print("\n" + "="*60)
    print(f"ROI ANALYSIS ({holding_period}-YEAR HOLDING PERIOD)")
    print("="*60)
    
    properties_affordable = df_metrics[df_metrics['ActualPrice'] <= investment_amount].copy()
    
    if len(properties_affordable) == 0:
        print(f"No properties found within ${investment_amount:,} budget")
        return
    
    properties_affordable['AppreciatedValue'] = (
        properties_affordable['ActualPrice'] * (1.03 ** holding_period)
    )
    
    properties_affordable['TotalRentalIncome'] = (
        properties_affordable['AnnualRentIncome'] * holding_period
    )
    
    total_expenses = (properties_affordable['AnnualTaxes'] + 
                     properties_affordable['MonthlyUtilityCost'] * 12) * holding_period
    
    properties_affordable['NetProfit'] = (
        properties_affordable['AppreciatedValue'] + 
        properties_affordable['TotalRentalIncome'] - 
        total_expenses - 
        properties_affordable['ActualPrice']
    ).round(0)
    
    properties_affordable['ROI_Percentage'] = (
        (properties_affordable['NetProfit'] / properties_affordable['ActualPrice'] * 100)
    ).round(2)
    
    top_roi = properties_affordable.nlargest(10, 'ROI_Percentage')[
        ['PropertyID', 'Location', 'ActualPrice', 'TotalRentalIncome', 
         'AppreciatedValue', 'NetProfit', 'ROI_Percentage']
    ]
    
    print(f"\nTop ROI Properties (Budget: ${investment_amount:,}):")
    print(top_roi.to_string(index=False))

def investment_report(df_metrics):
    """Generate comprehensive investment report"""
    print("\n" + "="*60)
    print("INVESTMENT SUMMARY REPORT")
    print("="*60)
    
    print(f"\nTotal Properties Analyzed: {len(df_metrics)}")
    print(f"Total Market Value: ${df_metrics['ActualPrice'].sum():,.0f}")
    print(f"Average Property Price: ${df_metrics['ActualPrice'].mean():,.0f}")
    
    print(f"\nYield Statistics:")
    print(f"  Average Gross Yield: {df_metrics['GrossYield'].mean():.2f}%")
    print(f"  Average Net Yield: {df_metrics['NetYield'].mean():.2f}%")
    print(f"  Max Yield: {df_metrics['GrossYield'].max():.2f}%")
    print(f"  Min Yield: {df_metrics['GrossYield'].min():.2f}%")
    
    print(f"\nPayback Analysis:")
    print(f"  Average Payback Period: {df_metrics['PaybackPeriod'].mean():.1f} years")
    print(f"  Shortest Payback: {df_metrics['PaybackPeriod'].min():.1f} years")
    print(f"  Longest Payback: {df_metrics['PaybackPeriod'].max():.1f} years")
    
    print(f"\nInvestment Score Distribution:")
    print(f"  Excellent (>80): {len(df_metrics[df_metrics['InvestmentScore'] > 80])} properties")
    print(f"  Good (60-80): {len(df_metrics[(df_metrics['InvestmentScore'] >= 60) & (df_metrics['InvestmentScore'] <= 80)])} properties")
    print(f"  Average (40-60): {len(df_metrics[(df_metrics['InvestmentScore'] >= 40) & (df_metrics['InvestmentScore'] < 60)])} properties")
    print(f"  Below Average (<40): {len(df_metrics[df_metrics['InvestmentScore'] < 40])} properties")

# ======================================================================
# SECTION 5: VISUALIZATIONS
# ======================================================================

def create_visualizations(df_metrics):
    """Create all professional visualizations"""
    print("\n[OK] Creating visualizations...")
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)
    
    # 1. Price Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_metrics['ActualPrice'], bins=30, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Property Price ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Property Price Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    df_metrics.boxplot(column='ActualPrice', by='Location', ax=axes[1])
    axes[1].set_xlabel('Location', fontsize=12)
    axes[1].set_ylabel('Property Price ($)', fontsize=12)
    axes[1].set_title('Price Distribution by Location', fontsize=14, fontweight='bold')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('01_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 01_price_distribution.png")
    
    # 2. Correlation Heatmap
    numeric_cols = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Garage', 
                   'ActualPrice', 'GrossYield', 'InvestmentScore', 'PaybackPeriod']
    correlation_matrix = df_metrics[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 02_correlation_heatmap.png")
    
    # 3. Yield Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df_metrics['GrossYield'], bins=25, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Gross Yield (%)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Gross Yield Distribution', fontsize=12, fontweight='bold')
    
    location_yield = df_metrics.groupby('Location')['GrossYield'].mean().sort_values(ascending=False)
    axes[0, 1].barh(location_yield.index, location_yield.values, color='teal')
    axes[0, 1].set_xlabel('Average Gross Yield (%)', fontsize=11)
    axes[0, 1].set_title('Average Yield by Location', fontsize=12, fontweight='bold')
    
    axes[1, 0].hist(df_metrics['PaybackPeriod'], bins=25, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Payback Period (years)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Payback Period Distribution', fontsize=12, fontweight='bold')
    
    scatter = axes[1, 1].scatter(df_metrics['PaybackPeriod'], df_metrics['GrossYield'], 
                                c=df_metrics['ActualPrice'], cmap='viridis', s=50, alpha=0.6)
    axes[1, 1].set_xlabel('Payback Period (years)', fontsize=11)
    axes[1, 1].set_ylabel('Gross Yield (%)', fontsize=11)
    axes[1, 1].set_title('Yield vs Payback Period', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1, 1], label='Price ($)')
    
    plt.tight_layout()
    plt.savefig('03_yield_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 03_yield_analysis.png")
    
    # 4. Location Comparison
    location_stats = df_metrics.groupby('Location').agg({
        'ActualPrice': 'mean',
        'SquareFeet': 'mean',
        'GrossYield': 'mean',
        'InvestmentScore': 'mean'
    }).sort_values('InvestmentScore', ascending=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    location_stats['ActualPrice'].plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Average Price by Location', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Average Price ($)', fontsize=11)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(alpha=0.3, axis='y')
    
    location_stats['GrossYield'].plot(kind='bar', ax=axes[0, 1], color='orange', alpha=0.7)
    axes[0, 1].set_title('Average Yield by Location', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Average Yield (%)', fontsize=11)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3, axis='y')
    
    location_stats['InvestmentScore'].plot(kind='bar', ax=axes[1, 0], color='red', alpha=0.7)
    axes[1, 0].set_title('Average Investment Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=11)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('04_location_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 04_location_comparison.png")
    
    # 5. Investment Score Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red' if x < 40 else 'orange' if x < 60 else 'lightgreen' if x < 80 else 'green' 
             for x in df_metrics['InvestmentScore']]
    
    ax.scatter(range(len(df_metrics)), df_metrics['InvestmentScore'].sort_values(ascending=False), 
              c=sorted(colors, reverse=True), s=50, alpha=0.6)
    
    ax.set_xlabel('Properties (sorted by score)', fontsize=12)
    ax.set_ylabel('Investment Score', fontsize=12)
    ax.set_title('Investment Score Distribution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('05_investment_score.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 05_investment_score.png")
    
    # 6. Property Type Analysis
    property_stats = df_metrics.groupby('PropertyType').agg({
        'ActualPrice': 'mean',
        'PropertyID': 'count',
        'GrossYield': 'mean'
    }).sort_values('ActualPrice', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    property_stats['PropertyID'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Number of Properties by Type', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.3, axis='y')
    
    property_stats['ActualPrice'].plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('Average Price by Property Type', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Average Price ($)', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.3, axis='y')
    
    property_stats['GrossYield'].plot(kind='bar', ax=axes[2], color='darkgreen')
    axes[2].set_title('Average Yield by Property Type', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Average Yield (%)', fontsize=11)
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('06_property_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 06_property_type_analysis.png")
    
    # 7. Price vs Size
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scatter = ax.scatter(df_metrics['SquareFeet'], df_metrics['ActualPrice'], 
                        c=df_metrics['GrossYield'], cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel('Square Feet', fontsize=12)
    ax.set_ylabel('Property Price ($)', fontsize=12)
    ax.set_title('Property Price vs Size (colored by Yield)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Gross Yield (%)', fontsize=11)
    
    z = np.polyfit(df_metrics['SquareFeet'], df_metrics['ActualPrice'], 1)
    p = np.poly1d(z)
    ax.plot(df_metrics['SquareFeet'].sort_values(), p(df_metrics['SquareFeet'].sort_values()), 
           "r--", alpha=0.8, linewidth=2, label='Trend line')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('07_price_vs_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 07_price_vs_size.png")
    
    # 8. Investment Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.6, f"${df_metrics['ActualPrice'].mean():,.0f}", 
            ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.text(0.5, 0.2, 'Avg Price', ha='center', va='center', fontsize=12)
    ax1.axis('off')
    ax1.set_facecolor('#f0f0f0')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.6, f"{df_metrics['GrossYield'].mean():.2f}%", 
            ha='center', va='center', fontsize=20, fontweight='bold', color='green')
    ax2.text(0.5, 0.2, 'Avg Yield', ha='center', va='center', fontsize=12)
    ax2.axis('off')
    ax2.set_facecolor('#f0f0f0')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.text(0.5, 0.6, f"{df_metrics['PaybackPeriod'].mean():.1f}y", 
            ha='center', va='center', fontsize=20, fontweight='bold', color='orange')
    ax3.text(0.5, 0.2, 'Avg Payback', ha='center', va='center', fontsize=12)
    ax3.axis('off')
    ax3.set_facecolor('#f0f0f0')
    
    ax4 = fig.add_subplot(gs[1, :2])
    location_count = df_metrics['Location'].value_counts()
    ax4.bar(location_count.index, location_count.values, color='steelblue')
    ax4.set_title('Properties by Location', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(alpha=0.3, axis='y')
    
    ax5 = fig.add_subplot(gs[1, 2])
    property_count = df_metrics['PropertyType'].value_counts()
    ax5.pie(property_count.values, labels=property_count.index, autopct='%1.1f%%')
    ax5.set_title('Property Type Distribution', fontweight='bold')
    
    ax6 = fig.add_subplot(gs[2, :])
    top_10 = df_metrics.nlargest(10, 'InvestmentScore')
    ax6.barh(range(len(top_10)), top_10['InvestmentScore'].values, color='darkgreen')
    ax6.set_yticks(range(len(top_10)))
    ax6.set_yticklabels([f"#{i+1}" for i in range(len(top_10))])
    ax6.set_xlabel('Investment Score')
    ax6.set_title('Top 10 Investment Opportunities', fontweight='bold')
    ax6.grid(alpha=0.3, axis='x')
    
    fig.suptitle('Real Estate Investment Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('08_investment_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: 08_investment_dashboard.png")

# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    """Execute complete analysis pipeline with command-line arguments"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Real Estate Investment Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python real_estate_analyzer.py
  python real_estate_analyzer.py --properties 1000 --budget 750000 --holding 15
  python real_estate_analyzer.py -p 300 -b 300000 -y 10
        '''
    )
    
    parser.add_argument('-p', '--properties', type=int, default=500,
                       help='Number of properties to generate (default: 500)')
    parser.add_argument('-b', '--budget', type=int, default=500000,
                       help='Investment budget in dollars (default: 500000)')
    parser.add_argument('-y', '--holding', type=int, default=10,
                       help='Holding period in years (default: 10)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    n_properties = args.properties
    budget = args.budget
    holding_period = args.holding
    create_viz = not args.no_viz
    
    print("\n" + "="*70)
    print(" "*15 + "REAL ESTATE INVESTMENT ANALYZER")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Properties: {n_properties}")
    print(f"  Budget: ${budget:,}")
    print(f"  Holding Period: {holding_period} years")
    print(f"  Visualizations: {'Enabled' if create_viz else 'Disabled'}")
    
    # Step 1: Generate Dataset
    print("\n[STEP 1] Generating Property Dataset...")
    properties = generate_property_dataset(n_properties=n_properties)
    properties.to_csv('property_data.csv', index=False)
    print(f"[OK] Generated {len(properties)} properties")
    print(f"[OK] Dataset saved to: property_data.csv")
    
    # Step 2: Data Cleaning & Analysis
    print("\n[STEP 2] Cleaning & Analyzing Data...")
    properties_clean = clean_data(properties)
    exploratory_analysis(properties_clean)
    correlation_analysis(properties_clean)
    location_analysis(properties_clean)
    
    # Step 3: Price Prediction
    print("\n[STEP 3] Building Price Prediction Models...")
    train_price_models(properties_clean)
    
    # Step 4: Investment Analysis
    print("\n[STEP 4] Calculating Investment Metrics...")
    df_metrics = calculate_investment_metrics(properties_clean)
    
    print("\n[STEP 5] Identifying Investment Opportunities...")
    identify_opportunities(df_metrics)
    
    print("\n[STEP 6] ROI Analysis...")
    roi_analysis(df_metrics, investment_amount=budget, holding_period=holding_period)
    
    print("\n[STEP 7] Investment Summary...")
    investment_report(df_metrics)
    
    # Step 8: Create Visualizations (optional)
    if create_viz:
        print("\n[STEP 8] Creating Visualizations...")
        create_visualizations(df_metrics)
    else:
        print("\n[STEP 8] Skipped visualization generation (--no-viz flag)")
    
    # Summary
    print("\n" + "="*70)
    print(" "*20 + "ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  [OK] property_data.csv - Clean dataset with investment metrics")
    if create_viz:
        print("  [OK] 01_price_distribution.png - Price analysis")
        print("  [OK] 02_correlation_heatmap.png - Feature correlations")
        print("  [OK] 03_yield_analysis.png - Yield & payback analysis")
        print("  [OK] 04_location_comparison.png - Location insights")
        print("  [OK] 05_investment_score.png - Investment score ranking")
        print("  [OK] 06_property_type_analysis.png - Property type comparison")
        print("  [OK] 07_price_vs_size.png - Price-size relationship")
        print("  [OK] 08_investment_dashboard.png - Executive dashboard")
    
    print("\nKey Insights:")
    print(f"  - Best Yielding Property: {df_metrics.loc[df_metrics['GrossYield'].idxmax(), 'PropertyID']}")
    print(f"    (Yield: {df_metrics['GrossYield'].max():.2f}%)")
    print(f"  - Best Investment Opportunity: {df_metrics.loc[df_metrics['InvestmentScore'].idxmax(), 'PropertyID']}")
    print(f"    (Score: {df_metrics['InvestmentScore'].max():.2f})")
    print(f"  - Average Investment Return: {df_metrics['CapRate'].mean():.2f}%")
    print(f"  - Average Payback Period: {df_metrics['PaybackPeriod'].mean():.1f} years")
    print(f"  - Properties within budget (${budget:,}): {len(df_metrics[df_metrics['ActualPrice'] <= budget])}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
