# Data Directory

This directory contains the raw and processed data for the asset pricing and investor sentiment analysis.

## Directory Structure

```
data/
├── raw/                # Raw data files (not committed to GitHub due to size)
│   ├── sentiment/      # Sentiment indicator data files
│   ├── factors/        # Factor portfolio data files
│   └── macro/          # Macroeconomic data files
├── processed/          # Processed/cleaned data files
└── README.md           # This file
```

## Data Sources

### Sentiment Indicators

#### 1. Baker-Wurgler Sentiment Index
- **Source**: Jeffrey Wurgler's website
- **URL**: http://people.stern.nyu.edu/jwurgler/
- **File**: `BW_Sentiment.xlsx`
- **Description**: Composite sentiment index based on six market-based proxies
- **Frequency**: Monthly (End-of-Month and End-of-Year versions)
- **Citation**: Baker, M., & Wurgler, J. (2006). Investor Sentiment and the Cross-Section of Stock Returns. *Journal of Finance*, 61(4), 1645-1680.

#### 2. Zhou et al. Sentiment Measures
- **Source**: Zhou et al. research
- **Files**:
  - `Zhou_InvestorSentiment.xlsx`
  - `Zhou_ManagerSentiment.xlsx`
  - `Zhou_EmployeeSentiment.xlsx`
- **Description**: Survey-based sentiment measures from different economic agents
- **Frequency**: Varies by measure
- **Citation**: Zhou, G. (2018). Measuring Investor Sentiment. *Annual Review of Financial Economics*, 10, 239-259.

#### 3. University of Michigan Consumer Sentiment
- **Source**: University of Michigan Surveys of Consumers
- **URL**: http://www.sca.isr.umich.edu/
- **Files**:
  - `UniMichigan_ConsumerSentiment.csv`
  - `UniMichigan_ExpectationsAndConditions.csv`
- **Components**:
  - ICS: Index of Consumer Sentiment
  - ICC: Index of Consumer Conditions
  - ICE: Index of Consumer Expectations
- **Frequency**: Monthly
- **Citation**: University of Michigan Surveys of Consumers

#### 4. AAII Sentiment Survey
- **Source**: American Association of Individual Investors
- **URL**: https://www.aaii.com/sentimentsurvey
- **File**: `AAII_sentiment.xlsx`
- **Description**: Bull-Bear Spread from weekly survey of individual investors
- **Frequency**: Weekly (aggregated to monthly)
- **Note**: Requires AAII membership for historical data

#### 5. CBOE Volatility Index (VIX)
- **Source**: Chicago Board Options Exchange via Federal Reserve Economic Data (FRED)
- **URL**: https://fred.stlouisfed.org/series/VIXCLS
- **File**: `CBoe_VIX.csv`
- **Description**: Market-implied volatility, used as inverse sentiment proxy
- **Frequency**: Daily (end-of-month values used)
- **Citation**: CBOE

### Factor Portfolios

#### 1. Open Asset Pricing
- **Source**: Open Asset Pricing library (Python package)
- **Installation**: `pip install openassetpricing`
- **Files**: Downloaded programmatically or from library website
- **Description**: Decile portfolios (value-weighted) for various anomalies across countries
- **Coverage**: Multiple countries and regions
- **Frequency**: Monthly
- **Citation**: Open Source Asset Pricing, Chen, A. and Zimmermann, T. (2022)

#### 2. JKP Factor Library
- **Source**: Jensen, Kelly, and Pedersen factor library
- **Files**:
  - `[all_countries]_[all_factors]_[monthly]_[vw].csv`
  - `[all_countries]_[all_themes]_[monthly]_[vw].csv`
  - `[all_regions]_[all_factors]_[monthly]_[vw].csv`
  - `[all_regions]_[all_themes]_[monthly]_[vw].csv`
  - `Factor Mapping.xlsx` (factor name mappings)
- **Description**: Comprehensive factor and theme portfolios
- **Frequency**: Monthly
- **Citation**: Contact data provider for specific citation

### Risk Factors

#### Fama-French Factors
- **Source**: Kenneth French Data Library
- **URL**: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- **Downloaded via**: pandas-datareader
- **Factors**:
  - **3-Factor Model**: Market (Mkt-RF), Size (SMB), Value (HML)
  - **5-Factor Model**: + Profitability (RMW), Investment (CMA)
  - **Risk-Free Rate**: RF (1-month Treasury bill)
- **Frequency**: Monthly
- **Citation**:
  - Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3-56.
  - Fama, E. F., & French, K. R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1-22.

### Macroeconomic Variables

Included in Baker-Wurgler sentiment data file:

1. **INDPRO**: Industrial Production Index
2. **DURABLE**: Durable Consumption
3. **NONDUR**: Nondurable Consumption
4. **SERVICES**: Services Consumption
5. **EMPLOY**: Employment
6. **CPI**: Consumer Price Index

All variables transformed to 12-month growth rates:
- Real variables (INDPRO, DURABLE, NONDUR, SERVICES, EMPLOY): Log-differences
- Nominal variables (CPI): Percentage changes

## Data Dictionary

### Processed Data Files

#### `orthogonalized_sentiment.csv`
| Column | Description | Unit |
|--------|-------------|------|
| date | Month-end date | YYYY-MM-DD |
| BW_orth | Orthogonalized Baker-Wurgler sentiment | Standardized |
| Zhou_Investor_orth | Orthogonalized investor sentiment | Standardized |
| ICS_orth | Orthogonalized consumer sentiment | Standardized |
| AAII_orth | Orthogonalized AAII bull-bear spread | Standardized |
| VIX_orth | Orthogonalized VIX | Standardized |
| Manager_orth | Orthogonalized manager sentiment | Standardized |
| Employee_orth | Orthogonalized employee sentiment | Standardized |

#### `factor_returns.csv`
| Column | Description | Unit |
|--------|-------------|------|
| date | Month-end date | YYYY-MM-DD |
| {factor}_Long | Decile 10 portfolio return | Decimal (0.05 = 5%) |
| {factor}_Short | Decile 01 portfolio return | Decimal |
| {factor}_Long-Short | Long minus short return | Decimal |
| country | Country code (if applicable) | ISO code |
| region | Region code (if applicable) | String |

#### `sentiment_conditional_results.csv`
| Column | Description | Unit |
|--------|-------------|------|
| sentiment | Sentiment indicator name | String |
| factor | Factor/anomaly name | String |
| portfolio_leg | Long/Short/Long-Short | String |
| High | Mean return in high sentiment | Decimal |
| Low | Mean return in low sentiment | Decimal |
| HML | High minus Low difference | Decimal |
| HML_tstat | Newey-West t-statistic | Float |
| significant | Indicator if |t| > 1.96 | Boolean |

## Data Usage Notes

### Alignment and Frequency
- All data aligned to month-end frequency
- Time series gaps identified and handled via `find_continuous_start_end_sentiment()`
- Only continuous periods used for analysis

### Missing Data
- Sentiment indicators: Use longest continuous period
- Factor portfolios: Remove signals with <36 continuous months
- Handle missing values before merging datasets

### Data Transformations

1. **Sentiment Orthogonalization**:
   ```
   Sentiment_t = α + Σ β_i * MacroGrowth_i,t + ε_t
   Orthogonalized Sentiment = ε_t
   ```

2. **Excess Returns**:
   ```
   Excess Return = Raw Return - Risk-Free Rate
   ```

3. **Benchmark-Adjusted Returns**:
   ```
   Residual_t = Return_t - (α + Σ β_j * Factor_j,t)
   ```

4. **Long-Short Sign Convention**:
   - Flipped if mean Long-Short < 0
   - Ensures consistent interpretation (positive = profitable)

## Downloading Data

### Automatic Downloads (via scripts)

```bash
# Download Fama-French factors
python scripts/download_data.py --source fama_french

# Download Open Asset Pricing data
python scripts/download_data.py --source oap
```

### Manual Downloads Required

1. **Baker-Wurgler Sentiment**: Download from Jeffrey Wurgler's website
2. **Zhou Sentiment Measures**: Obtain from authors or replication materials
3. **University of Michigan**: Available via FRED or directly from UM
4. **AAII Sentiment**: Requires AAII membership
5. **JKP Factors**: Contact data provider

## Data Size Considerations

**Large Files (not committed to GitHub):**
- `full_combined_dataset.csv`: ~87 MB
- `df_pivoted_residuals.csv`: ~194 MB
- `df_monthly_residuals.csv`: ~265 MB

**Storage Recommendations:**
- Keep raw data in `data/raw/`
- Store processed checkpoints in `results/checkpoints/`
- Use `.gitignore` to prevent accidental commits

## Data Update Frequency

| Data Source | Update Frequency | Last Updated |
|-------------|------------------|--------------|
| Baker-Wurgler | Annual | Check website |
| Fama-French | Monthly | Auto-updated via API |
| Open Asset Pricing | Periodic | Check library updates |
| University of Michigan | Monthly | Via FRED |
| VIX | Daily | Via FRED |
| AAII | Weekly | Via AAII |

## License and Attribution

When using this data:

1. **Cite original sources** (see citations above)
2. **Respect data licenses** - some data requires membership or permission
3. **Check usage rights** - academic vs commercial use may differ
4. **Attribute properly** in any publications or presentations

## Contact

For data access issues or questions:
- Check original source websites
- Review data provider documentation
- Contact data providers directly for access

## Disclaimer

This repository does not redistribute proprietary data. Users must obtain data from original sources following their terms of use.
