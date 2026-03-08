# Methodology

This document describes the econometric methodology used in the asset pricing and investor sentiment analysis.

## Overview

The analysis investigates whether investor sentiment—once purified of macroeconomic confounds—can explain or predict returns of well-documented factor portfolios. The methodology proceeds in three main stages:

1. Sentiment Orthogonalization
2. Factor Model Testing
3. Sentiment-Conditional Analysis

---

## 1. Sentiment Orthogonalization

### Motivation

Raw sentiment indicators capture both macroeconomic conditions and behavioral/psychological components. Since macroeconomic conditions also explain factor returns, conflating the two would lead to biased inferences about the behavioral channel.

### Procedure

For each sentiment indicator $S_t$, we estimate:

$$S_t = \alpha + \sum_{k=1}^{K} \beta_k \Delta^{12} X_{k,t} + \varepsilon_t$$

Where:
- $\Delta^{12} X_{k,t}$ = 12-month growth rate of macroeconomic variable $k$
- $\varepsilon_t$ = residuals = **orthogonalized (behavioral) sentiment**

### Macroeconomic Variables

| Variable | Transformation | Source |
|----------|---------------|--------|
| Industrial Production | 12-month log-difference | BW dataset |
| Durable Consumption | 12-month log-difference | BW dataset |
| Nondurable Consumption | 12-month log-difference | BW dataset |
| Services | 12-month log-difference | BW dataset |
| Employment | 12-month log-difference | BW dataset |
| CPI | 12-month percentage change | BW dataset |

**Log-difference formula (real variables):**
$$\Delta^{12} X_t = \ln(X_t) - \ln(X_{t-12})$$

**Percentage change formula (nominal variables):**
$$\Delta^{12} X_t = \frac{X_t - X_{t-12}}{X_{t-12}}$$

### Estimation Method

- **Model**: Ordinary Least Squares (OLS)
- **Standard Errors**: Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC)
- **Lags**: 4 (for orthogonalization regressions)

**Newey-West Variance Estimator:**

$$\hat{V}_{NW} = \hat{\Gamma}_0 + \sum_{l=1}^{L} \left(1 - \frac{l}{L+1}\right)(\hat{\Gamma}_l + \hat{\Gamma}_l')$$

Where $\hat{\Gamma}_l$ is the sample autocovariance at lag $l$ and $L$ is the truncation lag.

### Diagnostics

For each orthogonalization, we report:

1. **R-squared**: Proportion of sentiment variation explained by macro variables
2. **VIF** (Variance Inflation Factor): Detect multicollinearity among macro predictors
   - VIF > 10: Serious multicollinearity
   - VIF > 5: Potential concern
3. **Breusch-Pagan test**: Test for heteroskedasticity in residuals
4. **Durbin-Watson statistic**: Test for autocorrelation
5. **Jarque-Bera test**: Test residuals for normality

---

## 2. Factor Model Testing

### Motivation

Not all factor portfolios represent genuine anomalies—some may just reflect compensation for risk. We use three increasingly comprehensive asset pricing models to identify robust anomalies.

### Factor Models

**CAPM:**
$$R_{it} - R_{ft} = \alpha_i + \beta_{i,MKT}(R_{mt} - R_{ft}) + \varepsilon_{it}$$

**Fama-French 3-Factor:**
$$R_{it} - R_{ft} = \alpha_i + \beta_{i,MKT}(R_{mt} - R_{ft}) + \beta_{i,SMB}SMB_t + \beta_{i,HML}HML_t + \varepsilon_{it}$$

**Fama-French 5-Factor:**
$$R_{it} - R_{ft} = \alpha_i + \beta_{i,MKT}(R_{mt} - R_{ft}) + \beta_{i,SMB}SMB_t + \beta_{i,HML}HML_t + \beta_{i,RMW}RMW_t + \beta_{i,CMA}CMA_t + \varepsilon_{it}$$

Where:
- $R_{it}$: Portfolio return at time $t$
- $R_{ft}$: Risk-free rate
- $\alpha_i$: Abnormal return (alpha)
- $MKT$: Market excess return
- $SMB$: Small Minus Big (size factor)
- $HML$: High Minus Low (value factor)
- $RMW$: Robust Minus Weak (profitability factor)
- $CMA$: Conservative Minus Aggressive (investment factor)

### Alpha Identification

- **Alpha ($\alpha_i$)**: Represents average excess return not explained by factor exposures
- **Statistical significance**: t-statistic with Newey-West standard errors (4 lags)
- **Robustness criterion**: Require significance at 5% level across **all three models** (CAPM, FF3, FF5)

### Benchmark-Adjusted Returns

After identifying robust anomalies, we compute benchmark-adjusted returns using the FF5 model:

$$\tilde{R}_{it} = R_{it} - R_{ft} - (\hat{\alpha}_i + \hat{\beta}_{i,MKT}(R_{mt} - R_{ft}) + \hat{\beta}_{i,SMB}SMB_t + \hat{\beta}_{i,HML}HML_t + \hat{\beta}_{i,RMW}RMW_t + \hat{\beta}_{i,CMA}CMA_t)$$

These residuals $\tilde{R}_{it}$ represent the portion of returns not explained by systematic risk factors—the pure anomaly component used for sentiment analysis.

---

## 3. Sentiment-Conditional Analysis

### Regime Classification

For each orthogonalized sentiment indicator, we classify months into regimes:
- **Low Sentiment**: Bottom tercile (below 33rd percentile)
- **High Sentiment**: Top tercile (above 67th percentile)
- **Mid Sentiment**: Middle tercile (excluded from analysis)

### Conditional Returns

For each factor $i$ and sentiment indicator $s$:

$$\bar{R}_{i}^{High} = \frac{1}{T_{High}} \sum_{t \in High_s} \tilde{R}_{it}$$

$$\bar{R}_{i}^{Low} = \frac{1}{T_{Low}} \sum_{t \in Low_s} \tilde{R}_{it}$$

### High-Minus-Low Test

The central test is:

$$HML_{i,s} = \bar{R}_{i}^{High} - \bar{R}_{i}^{Low}$$

**Statistical Significance**: Newey-West t-statistic with 3 lags:

$$t_{NW} = \frac{HML_{i,s}}{\hat{SE}_{NW}(HML_{i,s})}$$

For testing whether $HML$ differs significantly from zero, we use:
- t > 2.576: Significant at 1% level (***)
- t > 1.96: Significant at 5% level (**)
- t > 1.645: Significant at 10% level (*)

### Portfolio Legs

Analysis conducted separately for:
1. **Long portfolios** (Decile 10): The high-characteristic portfolio
2. **Short portfolios** (Decile 01): The low-characteristic portfolio
3. **Long-Short portfolios** (10 - 01): Full zero-investment strategy

**Sign Convention**: Long-Short returns are flipped if the unconditional mean is negative, ensuring consistent interpretation (positive = profitable anomaly).

---

## 4. Supplementary Analyses

### Correlation Analysis

Pairwise Pearson correlations between sentiment indicators across multiple sample periods:
- Full Sample
- Long Sample (1965–present)
- Modern Sample (1990–present)
- Manager Window (available manager sentiment period)
- Full Overlap (common sample across all indicators)

**Multiple Testing Correction**: Benjamini-Hochberg False Discovery Rate (FDR) procedure at 5% level.

### Principal Component Analysis

PCA is applied to standardized sentiment indicators to:
1. Identify the common behavioral factor
2. Assess collinearity among indicators
3. Construct a composite sentiment measure

**Procedure**:
1. Standardize each indicator: $z_{it} = \frac{S_{it} - \bar{S}_i}{\sigma_{S_i}}$
2. Run PCA on standardized indicators
3. Extract loadings, explained variance, and principal component scores

PCA is run separately for each sample period to assess stability.

---

## Statistical Reference

### Significance Levels

| Symbol | Level | t-threshold (one-sided) | t-threshold (two-sided) |
|--------|-------|------------------------|------------------------|
| * | 10% | 1.282 | 1.645 |
| ** | 5% | 1.645 | 1.960 |
| *** | 1% | 2.326 | 2.576 |

### Newey-West Lags Summary

| Application | Lags | Reason |
|-------------|------|--------|
| Sentiment orthogonalization | 4 | Quarterly dynamics in sentiment |
| Benchmark-adjusted returns | 8 | Annual dynamics in residuals |
| HML t-tests | 3 | Quarterly dynamics in factor returns |

---

## References

- Baker, M., & Wurgler, J. (2006). Investor Sentiment and the Cross-Section of Stock Returns. *Journal of Finance*, 61(4), 1645–1680.
- Fama, E. F., & French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Fama, E. F., & French, K. R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1), 1–22.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Zhou, G. (2018). Measuring Investor Sentiment. *Annual Review of Financial Economics*, 10, 239–259.
