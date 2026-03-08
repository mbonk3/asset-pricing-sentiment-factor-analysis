"""
Regression Utility Functions

This module provides regression utilities including OLS with Newey-West standard errors,
residual calculation, and result parsing for asset pricing models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResults


def run_ols_newey_west(
    y: Union[pd.Series, np.ndarray],
    X: Union[pd.DataFrame, np.ndarray],
    lags: int = 4,
    add_constant: bool = True
) -> RegressionResults:
    """
    Run OLS regression with Newey-West HAC standard errors.

    Performs Ordinary Least Squares regression with Heteroskedasticity and
    Autocorrelation Consistent (HAC) standard errors using the Newey-West method.

    Args:
        y: Dependent variable
        X: Independent variables (can be DataFrame or array)
        lags: Number of lags for Newey-West adjustment
        add_constant: Whether to add constant/intercept to X

    Returns:
        Regression results with Newey-West covariance

    Examples:
        >>> y = pd.Series(np.random.normal(0, 1, 100))
        >>> X = pd.DataFrame({'x1': np.random.normal(0, 1, 100)})
        >>> results = run_ols_newey_west(y, X, lags=4)
        >>> hasattr(results, 'params')
        True
    """
    # Convert to DataFrame/Series if needed
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    # Add constant if requested
    if add_constant:
        X = sm.add_constant(X)

    # Remove missing values
    data = pd.concat([y, X], axis=1).dropna()
    y_clean = data.iloc[:, 0]
    X_clean = data.iloc[:, 1:]

    # Run OLS
    model = OLS(y_clean, X_clean)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    return results


def get_residuals(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    lags: int = 8
) -> pd.Series:
    """
    Calculate regression residuals with Newey-West adjustment.

    Regresses y on factors and returns residuals (benchmark-adjusted returns).
    These residuals represent returns not explained by the factor model.

    Args:
        df: DataFrame containing y and factors
        y_col: Name of dependent variable column
        factor_cols: List of factor column names
        lags: Number of lags for Newey-West

    Returns:
        Series of residuals

    Examples:
        >>> df = pd.DataFrame({
        ...     'returns': np.random.normal(0.5, 1, 100),
        ...     'mkt': np.random.normal(0, 1, 100),
        ...     'smb': np.random.normal(0, 1, 100)
        ... })
        >>> residuals = get_residuals(df, 'returns', ['mkt', 'smb'], lags=4)
        >>> len(residuals) <= len(df)
        True
    """
    # Prepare data
    y = df[y_col]
    X = df[factor_cols]

    # Run regression
    results = run_ols_newey_west(y, X, lags=lags, add_constant=True)

    # Return residuals
    return results.resid


def extract_alpha_tstats(
    results: RegressionResults
) -> Dict[str, float]:
    """
    Extract alpha (intercept) and t-statistic from regression results.

    Args:
        results: Statsmodels regression results

    Returns:
        Dictionary with alpha, t-statistic, and p-value

    Examples:
        >>> y = pd.Series(np.random.normal(0, 1, 100))
        >>> X = pd.DataFrame({'x1': np.random.normal(0, 1, 100)})
        >>> results = run_ols_newey_west(y, X)
        >>> alpha_dict = extract_alpha_tstats(results)
        >>> 'alpha' in alpha_dict
        True
    """
    try:
        # Alpha is the constant/intercept
        alpha = results.params['const'] if 'const' in results.params else results.params.iloc[0]

        # T-statistic
        tstat = results.tvalues['const'] if 'const' in results.tvalues else results.tvalues.iloc[0]

        # P-value
        pvalue = results.pvalues['const'] if 'const' in results.pvalues else results.pvalues.iloc[0]

        # R-squared
        rsquared = results.rsquared

        return {
            'alpha': alpha,
            'tstat': tstat,
            'pvalue': pvalue,
            'rsquared': rsquared
        }
    except Exception as e:
        return {
            'alpha': np.nan,
            'tstat': np.nan,
            'pvalue': np.nan,
            'rsquared': np.nan,
            'error': str(e)
        }


def regression_summary_dict(
    results: RegressionResults
) -> Dict[str, Union[float, int]]:
    """
    Create a comprehensive dictionary of regression results.

    Args:
        results: Statsmodels regression results

    Returns:
        Dictionary with regression statistics

    Examples:
        >>> y = pd.Series(np.random.normal(0, 1, 100))
        >>> X = pd.DataFrame({'x1': np.random.normal(0, 1, 100)})
        >>> results = run_ols_newey_west(y, X)
        >>> summary_dict = regression_summary_dict(results)
        >>> 'nobs' in summary_dict
        True
    """
    summary = {
        'nobs': int(results.nobs),
        'rsquared': results.rsquared,
        'rsquared_adj': results.rsquared_adj,
        'fvalue': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'aic': results.aic,
        'bic': results.bic,
        'durbin_watson': sm.stats.stattools.durbin_watson(results.resid)
    }

    # Add coefficients, t-stats, and p-values
    for var in results.params.index:
        summary[f'{var}_coef'] = results.params[var]
        summary[f'{var}_tstat'] = results.tvalues[var]
        summary[f'{var}_pvalue'] = results.pvalues[var]

    return summary


def rolling_regression(
    df: pd.DataFrame,
    y_col: str,
    X_cols: List[str],
    window: int,
    lags: int = 4
) -> pd.DataFrame:
    """
    Perform rolling window regressions.

    Estimates regression coefficients using a rolling window, useful for
    analyzing time-varying relationships.

    Args:
        df: DataFrame with y and X variables
        y_col: Dependent variable column name
        X_cols: Independent variable column names
        window: Rolling window size in periods
        lags: Lags for Newey-West

    Returns:
        DataFrame with rolling coefficients

    Examples:
        >>> df = pd.DataFrame({
        ...     'y': np.random.normal(0, 1, 200),
        ...     'x': np.random.normal(0, 1, 200)
        ... })
        >>> rolling_coefs = rolling_regression(df, 'y', ['x'], window=50)
        >>> 'const' in rolling_coefs.columns
        True
    """
    results_list = []

    for i in range(window, len(df) + 1):
        # Get window data
        window_data = df.iloc[i - window:i]

        # Run regression
        try:
            y = window_data[y_col]
            X = window_data[X_cols]

            reg_results = run_ols_newey_west(y, X, lags=lags)

            # Store coefficients with date
            coef_dict = {'date': window_data.index[-1]}
            coef_dict.update(reg_results.params.to_dict())

            results_list.append(coef_dict)

        except Exception as e:
            # Skip if regression fails
            continue

    # Convert to DataFrame
    rolling_coefs = pd.DataFrame(results_list)

    if 'date' in rolling_coefs.columns:
        rolling_coefs = rolling_coefs.set_index('date')

    return rolling_coefs


def grouped_regression(
    df: pd.DataFrame,
    y_col: str,
    X_cols: List[str],
    group_col: str,
    lags: int = 4
) -> pd.DataFrame:
    """
    Run regressions separately for each group.

    Useful for analyzing cross-sectional differences (e.g., by country,
    by time period, etc.).

    Args:
        df: DataFrame with y, X, and grouping variable
        y_col: Dependent variable column name
        X_cols: Independent variable column names
        group_col: Grouping column name
        lags: Lags for Newey-West

    Returns:
        DataFrame with regression results for each group

    Examples:
        >>> df = pd.DataFrame({
        ...     'y': np.random.normal(0, 1, 200),
        ...     'x': np.random.normal(0, 1, 200),
        ...     'group': ['A'] * 100 + ['B'] * 100
        ... })
        >>> group_results = grouped_regression(df, 'y', ['x'], 'group')
        >>> len(group_results) >= 1
        True
    """
    results_list = []

    for group_name, group_data in df.groupby(group_col):
        try:
            # Run regression for this group
            y = group_data[y_col]
            X = group_data[X_cols]

            reg_results = run_ols_newey_west(y, X, lags=lags)

            # Extract results
            result_dict = {
                group_col: group_name,
                'nobs': int(reg_results.nobs)
            }

            # Add coefficients
            for var in reg_results.params.index:
                result_dict[f'{var}_coef'] = reg_results.params[var]
                result_dict[f'{var}_tstat'] = reg_results.tvalues[var]
                result_dict[f'{var}_pvalue'] = reg_results.pvalues[var]

            result_dict['rsquared'] = reg_results.rsquared

            results_list.append(result_dict)

        except Exception as e:
            # Log error but continue
            print(f"Regression failed for group {group_name}: {e}")
            continue

    return pd.DataFrame(results_list)


def calculate_regression_diagnostics(
    results: RegressionResults
) -> Dict[str, float]:
    """
    Calculate comprehensive regression diagnostics.

    Args:
        results: Statsmodels regression results

    Returns:
        Dictionary with diagnostic statistics

    Examples:
        >>> y = pd.Series(np.random.normal(0, 1, 100))
        >>> X = pd.DataFrame({'x1': np.random.normal(0, 1, 100)})
        >>> results = run_ols_newey_west(y, X)
        >>> diagnostics = calculate_regression_diagnostics(results)
        >>> 'durbin_watson' in diagnostics
        True
    """
    diagnostics = {
        'durbin_watson': sm.stats.stattools.durbin_watson(results.resid),
        'jarque_bera': sm.stats.stattools.jarque_bera(results.resid)[0],
        'jarque_bera_pvalue': sm.stats.stattools.jarque_bera(results.resid)[1],
        'omnibus': sm.stats.stattools.omni_normtest(results.resid)[0],
        'omnibus_pvalue': sm.stats.stattools.omni_normtest(results.resid)[1],
        'condition_number': results.condition_number
    }

    return diagnostics
