# Asset Pricing and Investor Sentiment Analysis

A comprehensive empirical study investigating the relationship between investor sentiment and factor-based portfolio returns, with a focus on orthogonalized sentiment measures and robust econometric methodology.

## Research Question

**Does investor sentiment—once purified of macroeconomic confounds—help explain or predict the returns of well-documented anomaly (factor-based) portfolios?**

This project examines whether behavioral sentiment components, isolated from macroeconomic fundamentals, have explanatory or predictive power for asset pricing anomalies after controlling for traditional risk factors (CAPM, Fama-French 3-factor, and Fama-French 5-factor models).

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project combines behavioral finance with quantitative asset pricing to investigate how investor sentiment affects anomaly returns. The analysis proceeds in three main stages:

1. **Sentiment Orthogonalization**: Purify 7 sentiment indicators of macroeconomic influences using OLS regression with Newey-West HAC standard errors
2. **Factor Model Analysis**: Test factor portfolios against CAPM, Fama-French 3-factor, and Fama-French 5-factor models to identify robust anomalies
3. **Sentiment-Conditional Analysis**: Examine how factor returns vary across high and low sentiment regimes using tercile-based classification

The project demonstrates advanced econometric techniques including:
- Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors
- Principal Component Analysis (PCA) for dimensionality reduction
- False Discovery Rate (FDR) correction for multiple testing
- Variance Inflation Factor (VIF) diagnostics for multicollinearity
- Regime-based conditional analysis with robust statistical testing

## Data Sources

### Sentiment Indicators (7 measures)

1. **Baker-Wurgler Sentiment Index** - Composite index based on market-based measures
2. **Zhou et al. Investor Sentiment** - Survey-based investor sentiment measure
3. **University of Michigan Consumer Sentiment (ICS)** - Consumer confidence index
4. **AAII Sentiment Survey** - Bull-Bear Spread from individual investors
5. **CBOE Volatility Index (VIX)** - Market-implied volatility (inverse sentiment proxy)
6. **Zhou et al. Manager Sentiment** - Corporate manager sentiment
7. **Zhou et al. Employee Sentiment** - Employee-based sentiment measure

### Factor Portfolios

- **Open Asset Pricing**: Decile portfolios (value-weighted) for various anomalies across countries
- **JKP Factor Library**: Factor and theme portfolios across regions
- **Kenneth French Data Library**: Risk factors (Market, SMB, HML, RMW, CMA) and risk-free rate

### Macroeconomic Variables (for orthogonalization)

- Industrial Production
- Durable Consumption
- Nondurable Consumption
- Services
- Employment
- Consumer Price Index (CPI)

All macro variables converted to 12-month growth rates for stationarity.

## Methodology

### 1. Sentiment Orthogonalization

Each sentiment indicator is regressed on macroeconomic growth rates to extract the behavioral component:

```
Sentiment_t = α + β₁(IP_growth_t) + β₂(Consumption_growth_t) + ... + ε_t
```

- **Residuals (ε_t)** represent orthogonalized sentiment (behavioral component)
- **Standard Errors**: Newey-West HAC adjustment (4-6 lags)
- **Diagnostics**: VIF for multicollinearity, Breusch-Pagan for heteroskedasticity

### 2. Factor Model Regressions

Test each anomaly portfolio against three asset pricing models:

**CAPM**:
```
R_it - R_ft = α + β(R_mt - R_ft) + ε_it
```

**Fama-French 3-Factor**:
```
R_it - R_ft = α + β_MKT(R_mt - R_ft) + β_SMB(SMB_t) + β_HML(HML_t) + ε_it
```

**Fama-French 5-Factor**:
```
R_it - R_ft = α + β_MKT(R_mt - R_ft) + β_SMB(SMB_t) + β_HML(HML_t) + β_RMW(RMW_t) + β_CMA(CMA_t) + ε_it
```

- **Alpha (α)**: Abnormal return not explained by risk factors
- **Residuals**: Benchmark-adjusted returns used for sentiment analysis
- **Filtering**: Require statistical significance across all three models for robustness

### 3. Sentiment-Conditional Analysis

Classify periods into High and Low sentiment regimes using tercile splits:
- **Low Sentiment**: Bottom 33% of orthogonalized sentiment
- **High Sentiment**: Top 33% of orthogonalized sentiment

For each factor portfolio:
1. Calculate mean returns conditional on sentiment regime
2. Compute High-Minus-Low (HML) difference
3. Test statistical significance using Newey-West t-statistics (3 lags)

Analyze separately for:
- Long portfolios (Decile 10)
- Short portfolios (Decile 01)
- Long-Short portfolios (10 - 01)

### 4. Statistical Techniques

- **PCA**: Reduce sentiment indicator dimensionality, assess common variation
- **FDR Correction**: Control false discovery rate in correlation analysis (Benjamini-Hochberg)
- **VIF**: Detect multicollinearity (threshold: 10)
- **Breusch-Pagan Test**: Test for heteroskedasticity
- **Newey-West HAC**: Robust standard errors for autocorrelation and heteroskedasticity

## Repository Structure

```
asset-pricing-sentiment-analysis/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
├── config/
│   └── config.yaml                     # Configuration parameters
├── data/
│   ├── raw/                            # Raw data files (not committed)
│   ├── processed/                      # Processed data outputs
│   └── README.md                       # Data dictionary and sources
├── src/                                # Source code modules
│   ├── data/                           # Data loading modules
│   │   ├── sentiment_loader.py         # Load sentiment indicators
│   │   ├── factor_loader.py            # Load factor portfolios
│   │   └── data_utils.py               # Data utilities
│   ├── preprocessing/                  # Data preprocessing
│   │   ├── orthogonalization.py        # Sentiment orthogonalization
│   │   ├── sentiment_processing.py     # Sentiment transformations
│   │   └── factor_processing.py        # Factor portfolio processing
│   ├── analysis/                       # Analysis modules
│   │   ├── factor_models.py            # CAPM, FF3, FF5 regressions
│   │   ├── sentiment_conditional.py    # Regime-based analysis
│   │   ├── statistical_tests.py        # Correlation, FDR, alpha survival
│   │   └── pca_analysis.py             # Principal component analysis
│   ├── utils/                          # Utility functions
│   │   ├── date_utils.py               # Date and time utilities
│   │   ├── statistical_utils.py        # Statistical functions
│   │   └── regression_utils.py         # Regression utilities
│   └── visualization/                  # Plotting and tables
│       ├── time_series_plots.py        # Time series visualizations
│       └── result_tables.py            # Result formatting and export
├── notebooks/                          # Jupyter notebooks
│   ├── 01_data_exploration.ipynb       # Data inspection and quality checks
│   ├── 02_sentiment_orthogonalization.ipynb  # Sentiment processing
│   ├── 03_factor_analysis.ipynb        # Factor model regressions
│   ├── 04_sentiment_conditional_analysis.ipynb  # Regime analysis
│   └── 05_main_analysis.ipynb          # Complete end-to-end workflow
├── scripts/                            # Executable scripts
│   ├── download_data.py                # Download public data sources
│   ├── process_sentiment.py            # Process sentiment indicators
│   ├── process_factors.py              # Process factor portfolios
│   └── run_full_analysis.py            # Complete pipeline execution
├── results/                            # Analysis outputs
│   ├── figures/                        # Plots and visualizations
│   ├── tables/                         # Result tables
│   └── checkpoints/                    # Intermediate checkpoints
├── tests/                              # Unit tests
│   ├── test_data_utils.py
│   ├── test_orthogonalization.py
│   └── test_regression_utils.py
└── docs/                               # Documentation
    ├── methodology.md                  # Detailed methodology
    ├── data_sources.md                 # Data sources and access
    └── results_interpretation.md       # How to interpret results
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/asset-pricing-sentiment-analysis.git
   cd asset-pricing-sentiment-analysis
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure data paths**:
   - Edit `config/config.yaml` to set data directory paths
   - Place raw data files in `data/raw/` (see [data/README.md](data/README.md) for details)

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python scripts/run_full_analysis.py
```

This will:
1. Load and process sentiment indicators
2. Orthogonalize sentiment against macro variables
3. Load and process factor portfolios
4. Run factor model regressions
5. Perform sentiment-conditional analysis
6. Generate tables and figures in `results/`

### Step-by-Step Analysis

Process individual components:

```bash
# Download publicly available data
python scripts/download_data.py

# Process sentiment indicators
python scripts/process_sentiment.py

# Process factor portfolios
python scripts/process_factors.py
```

### Interactive Analysis

Explore the analysis interactively using Jupyter notebooks:

```bash
jupyter notebook
```

Navigate to `notebooks/05_main_analysis.ipynb` for the complete workflow, or explore individual notebooks for detailed analysis stages.

### Using the Modules

Import and use analysis functions in your own code:

```python
from src.preprocessing.orthogonalization import orthogonalize_sentiment
from src.analysis.factor_models import run_fama_french_5factor
from src.analysis.sentiment_conditional import sentiment_factor_analysis

# Orthogonalize sentiment
orthogonalized = orthogonalize_sentiment(sentiment_series, macro_df, lags=4)

# Run FF5 regression
results = run_fama_french_5factor(excess_returns, ff5_factors)

# Sentiment-conditional analysis
conditional_results = sentiment_factor_analysis(sentiment_df, residuals_df)
```

## Key Findings

*Note: This section should be updated with actual findings from your analysis*

Key insights from the analysis:

1. **Sentiment Orthogonalization**: Orthogonalized sentiment indicators show [X% correlation reduction] with macroeconomic variables while preserving [Y%] of original variation

2. **Factor Model Results**: [Z factors] demonstrate significant alphas across all three models (CAPM, FF3, FF5), suggesting robust anomaly returns

3. **Sentiment-Conditional Returns**: Factor returns exhibit [direction] sensitivity to sentiment regimes, with High-Minus-Low differences of [X bps] per month on average

4. **PCA Insights**: First principal component explains [X%] of sentiment variation, suggesting [interpretation of common factor]

*Detailed results available in `results/tables/` and visualizations in `results/figures/`*

## Technologies Used

### Core Libraries

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **statsmodels**: Econometric models and statistical tests
- **scikit-learn**: Machine learning (PCA, preprocessing)
- **matplotlib** / **seaborn**: Data visualization

### Data Sources

- **pandas-datareader**: Fetch Fama-French factors
- **openassetpricing**: Access factor portfolio data

### Statistical Techniques

- **Newey-West HAC Standard Errors**: Robust inference under heteroskedasticity and autocorrelation
- **Principal Component Analysis**: Dimensionality reduction and common factor extraction
- **False Discovery Rate (FDR)**: Multiple testing correction
- **VIF and Breusch-Pagan**: Diagnostic testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

### Data Providers

- **Baker and Wurgler** for the sentiment index and macroeconomic data
- **Zhou et al.** for investor, manager, and employee sentiment measures
- **University of Michigan** for consumer sentiment data
- **AAII** for the sentiment survey
- **CBOE** for VIX data
- **Open Asset Pricing** for factor portfolio data
- **Kenneth French Data Library** for risk factors
- **JKP Factor Library** for additional factor data

### Academic References

Key papers informing this analysis:

1. Baker, M., & Wurgler, J. (2006). Investor Sentiment and the Cross-Section of Stock Returns. *Journal of Finance*, 61(4), 1645-1680.
2. Fama, E. F., & French, K. R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1-22.
3. Zhou, G. (2018). Measuring Investor Sentiment. *Annual Review of Financial Economics*, 10, 239-259.

---

**Maintainer**: Maximilian Bonk
**Last Updated**: February 2026
