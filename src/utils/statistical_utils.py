"""
Statistical Utility Functions

This module provides statistical functions for econometric analysis including
Newey-West t-statistics, VIF calculation, heteroskedasticity tests, and
multiple testing corrections.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from scipy import stats


def newey_west_tstat(series: pd.Series, lags: int = 3) -> float:
    """
    Compute Newey-West t-statistic for a return series.

    Calculates the t-statistic for testing if the mean of a return series
    is significantly different from zero, using Newey-West HAC standard errors
    to account for heteroskedasticity and autocorrelation.

    Args:
        series: Return series (e.g., High-Minus-Low returns)
        lags: Number of lags for Newey-West adjustment

    Returns:
        Newey-West t-statistic

    Examples:
        >>> returns = pd.Series(np.random.normal(0.5, 1, 100))
        >>> t_stat = newey_west_tstat(returns, lags=3)
        >>> isinstance(t_stat, float)
        True
    """
    # Remove missing values
    series = series.dropna()

    if len(series) == 0:
        return np.nan

    # Mean return
    mean_return = series.mean()

    # Number of observations
    n = len(series)

    if n <= lags:
        # Not enough observations for Newey-West
        # Fall back to simple t-stat
        if series.std() == 0:
            return np.nan
        return mean_return / (series.std() / np.sqrt(n))

    # Demeaned series
    demeaned = series - mean_return

    # Calculate variance with Newey-West adjustment
    # Gamma_0 (variance)
    gamma_0 = (demeaned ** 2).sum() / n

    # Autocovariances with Bartlett weights
    gamma_sum = 0
    for lag in range(1, lags + 1):
        # Bartlett weight
        weight = 1 - lag / (lags + 1)

        # Autocovariance at lag
        gamma_lag = (demeaned.iloc[lag:].values * demeaned.iloc[:-lag].values).sum() / n

        # Add weighted autocovariance (both sides)
        gamma_sum += 2 * weight * gamma_lag

    # Newey-West variance estimate
    nw_variance = gamma_0 + gamma_sum

    if nw_variance <= 0:
        return np.nan

    # Newey-West standard error
    nw_se = np.sqrt(nw_variance / n)

    if nw_se == 0:
        return np.nan

    # T-statistic
    t_stat = mean_return / nw_se

    return t_stat


def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.

    VIF measures how much the variance of a regression coefficient is inflated
    due to multicollinearity. VIF > 10 indicates problematic multicollinearity.

    Args:
        df: DataFrame containing the features
        features: List of column names to compute VIF for

    Returns:
        DataFrame with features and their VIF values

    Examples:
        >>> df = pd.DataFrame({
        ...     'x1': np.random.normal(0, 1, 100),
        ...     'x2': np.random.normal(0, 1, 100)
        ... })
        >>> df['x3'] = df['x1'] + df['x2']  # Multicollinear
        >>> vif_df = compute_vif(df, ['x1', 'x2', 'x3'])
        >>> 'VIF' in vif_df.columns
        True
    """
    # Subset to features only
    X = df[features].dropna()

    # Calculate VIF for each feature
    vif_data = []
    for i, feature in enumerate(features):
        try:
            vif_value = variance_inflation_factor(X.values, i)
            vif_data.append({'Feature': feature, 'VIF': vif_value})
        except Exception as e:
            vif_data.append({'Feature': feature, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data)

    return vif_df


def breusch_pagan_test(
    residuals: np.ndarray,
    exog: np.ndarray
) -> Dict[str, float]:
    """
    Perform Breusch-Pagan test for heteroskedasticity.

    Tests the null hypothesis that residuals have constant variance
    (homoskedasticity). Low p-value indicates heteroskedasticity.

    Args:
        residuals: Regression residuals
        exog: Exogenous variables (design matrix)

    Returns:
        Dictionary with test statistic and p-value

    Examples:
        >>> residuals = np.random.normal(0, 1, 100)
        >>> exog = np.random.normal(0, 1, (100, 2))
        >>> result = breusch_pagan_test(residuals, exog)
        >>> 'pvalue' in result
        True
    """
    try:
        lm_stat, lm_pvalue, fstat, f_pvalue = het_breuschpagan(residuals, exog)

        return {
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_statistic': fstat,
            'f_pvalue': f_pvalue
        }
    except Exception as e:
        return {
            'lm_statistic': np.nan,
            'lm_pvalue': np.nan,
            'f_statistic': np.nan,
            'f_pvalue': np.nan,
            'error': str(e)
        }


def fdr_correction(
    pvalues: Union[np.ndarray, pd.Series, List[float]],
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply False Discovery Rate (FDR) correction for multiple testing.

    Uses Benjamini-Hochberg procedure to control the expected proportion
    of false discoveries among rejected hypotheses.

    Args:
        pvalues: Array of p-values to correct
        alpha: Desired FDR level (default: 0.05)
        method: Correction method - 'fdr_bh' (Benjamini-Hochberg),
                'fdr_by' (Benjamini-Yekutieli), or 'bonferroni'

    Returns:
        Tuple of (rejected, corrected_pvalues) where rejected is boolean array
        indicating which hypotheses are rejected

    Examples:
        >>> pvalues = [0.001, 0.05, 0.1, 0.5]
        >>> rejected, corrected = fdr_correction(pvalues, alpha=0.05)
        >>> isinstance(rejected, np.ndarray)
        True
    """
    # Convert to numpy array
    pvalues_array = np.array(pvalues)

    # Apply correction
    rejected, corrected_pvalues, _, _ = multipletests(
        pvalues_array,
        alpha=alpha,
        method=method
    )

    return rejected, corrected_pvalues


def pearson_correlation_test(
    x: pd.Series,
    y: pd.Series
) -> Dict[str, float]:
    """
    Compute Pearson correlation and test for significance.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Dictionary with correlation coefficient and p-value

    Examples:
        >>> x = pd.Series(np.random.normal(0, 1, 100))
        >>> y = pd.Series(np.random.normal(0, 1, 100))
        >>> result = pearson_correlation_test(x, y)
        >>> 'correlation' in result
        True
    """
    # Align and remove missing
    df = pd.DataFrame({'x': x, 'y': y}).dropna()

    if len(df) < 3:
        return {'correlation': np.nan, 'pvalue': np.nan, 'n': len(df)}

    # Compute correlation and p-value
    corr, pvalue = stats.pearsonr(df['x'], df['y'])

    return {
        'correlation': corr,
        'pvalue': pvalue,
        'n': len(df)
    }


def winsorize(
    series: pd.Series,
    limits: Tuple[float, float] = (0.01, 0.99)
) -> pd.Series:
    """
    Winsorize a series at specified quantiles.

    Caps extreme values at specified percentiles to reduce the impact
    of outliers.

    Args:
        series: Data series to winsorize
        limits: Tuple of (lower_percentile, upper_percentile)

    Returns:
        Winsorized series

    Examples:
        >>> data = pd.Series([1, 2, 3, 100, 200])
        >>> winsorized = winsorize(data, limits=(0.2, 0.8))
        >>> winsorized.max() < 200
        True
    """
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(limits[1])

    return series.clip(lower=lower_limit, upper=upper_limit)


def standardize(series: pd.Series) -> pd.Series:
    """
    Standardize a series to have mean 0 and standard deviation 1.

    Args:
        series: Data series to standardize

    Returns:
        Standardized series (z-scores)

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> standardized = standardize(data)
        >>> abs(standardized.mean()) < 1e-10
        True
    """
    return (series - series.mean()) / series.std()


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for a DataFrame.

    Args:
        df: DataFrame with numeric columns

    Returns:
        DataFrame with descriptive statistics (count, mean, std, min, quartiles, max, skew, kurtosis)

    Examples:
        >>> df = pd.DataFrame({'a': range(100), 'b': range(100, 200)})
        >>> stats_df = descriptive_stats(df)
        >>> 'mean' in stats_df.index
        True
    """
    # Basic statistics
    desc = df.describe()

    # Add skewness and kurtosis
    desc.loc['skew'] = df.skew()
    desc.loc['kurt'] = df.kurt()

    return desc


def calculate_terciles(
    series: pd.Series,
    labels: List[str] = ['Low', 'Mid', 'High']
) -> pd.Series:
    """
    Split a series into terciles (three equal groups).

    Args:
        series: Data series to split
        labels: Labels for the three groups

    Returns:
        Categorical series with tercile labels

    Examples:
        >>> data = pd.Series(range(100))
        >>> terciles = calculate_terciles(data)
        >>> terciles.value_counts()['Low']
        33
    """
    return pd.qcut(series, q=3, labels=labels, duplicates='drop')
