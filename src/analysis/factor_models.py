"""
Factor Model Analysis Module

This module implements CAPM, Fama-French 3-factor, and Fama-French 5-factor
regressions for testing anomaly portfolio returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os
import pandas_datareader.data as web

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.regression_utils import (
    run_ols_newey_west,
    get_residuals,
    extract_alpha_tstats
)


def load_fama_french_3factor(
    start: str = '1960-01-01',
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Download Fama-French 3-factor data from Kenneth French's data library.

    Args:
        start: Start date in 'YYYY-MM-DD' format
        end: End date (defaults to today)

    Returns:
        DataFrame with columns: Mkt-RF, SMB, HML, RF

    Examples:
        >>> # ff3 = load_fama_french_3factor(start='2000-01-01')
        >>> # 'Mkt-RF' in ff3.columns
        >>> pass
    """
    ff3 = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=start, end=end)
    df = ff3[0].copy()

    # Convert from percent to decimal
    df = df / 100

    # Rename index to date
    df.index = pd.to_datetime(df.index.to_timestamp())

    return df


def load_fama_french_5factor(
    start: str = '1963-07-01',
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Download Fama-French 5-factor data from Kenneth French's data library.

    Args:
        start: Start date in 'YYYY-MM-DD' format
        end: End date (defaults to today)

    Returns:
        DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF

    Examples:
        >>> # ff5 = load_fama_french_5factor(start='2000-01-01')
        >>> # 'RMW' in ff5.columns
        >>> pass
    """
    ff5 = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start, end=end)
    df = ff5[0].copy()

    # Convert from percent to decimal
    df = df / 100

    # Rename index to date
    df.index = pd.to_datetime(df.index.to_timestamp())

    return df


def run_capm(
    excess_returns: pd.Series,
    market_excess: pd.Series,
    lags: int = 4
) -> Dict:
    """
    Run CAPM regression: R_it - R_ft = α + β(Mkt-RF) + ε

    Args:
        excess_returns: Portfolio excess returns (already minus RF)
        market_excess: Market excess return (Mkt-RF)
        lags: Newey-West lags

    Returns:
        Dictionary with alpha, t-stat, p-value, beta, and R²

    Examples:
        >>> excess_ret = pd.Series(np.random.normal(0.005, 0.05, 100))
        >>> mkt = pd.Series(np.random.normal(0.006, 0.04, 100))
        >>> result = run_capm(excess_ret, mkt)
        >>> 'alpha' in result
        True
    """
    X = pd.DataFrame({'Mkt-RF': market_excess})
    results = run_ols_newey_west(excess_returns, X, lags=lags, add_constant=True)
    alpha_info = extract_alpha_tstats(results)

    return {
        'model': 'CAPM',
        'alpha': alpha_info['alpha'],
        'alpha_tstat': alpha_info['tstat'],
        'alpha_pvalue': alpha_info['pvalue'],
        'beta_mkt': results.params.get('Mkt-RF', np.nan),
        'rsquared': results.rsquared,
        'nobs': int(results.nobs)
    }


def run_fama_french_3factor_model(
    excess_returns: pd.Series,
    ff3_factors: pd.DataFrame,
    lags: int = 4
) -> Dict:
    """
    Run Fama-French 3-factor regression: R_it - R_ft = α + β_MKT + β_SMB + β_HML + ε

    Args:
        excess_returns: Portfolio excess returns
        ff3_factors: DataFrame with Mkt-RF, SMB, HML columns
        lags: Newey-West lags

    Returns:
        Dictionary with alpha, t-stat, factor loadings, and R²

    Examples:
        >>> excess_ret = pd.Series(np.random.normal(0.005, 0.05, 100))
        >>> ff3 = pd.DataFrame({
        ...     'Mkt-RF': np.random.normal(0.006, 0.04, 100),
        ...     'SMB': np.random.normal(0, 0.03, 100),
        ...     'HML': np.random.normal(0, 0.03, 100)
        ... })
        >>> result = run_fama_french_3factor_model(excess_ret, ff3)
        >>> 'alpha' in result
        True
    """
    factor_cols = ['Mkt-RF', 'SMB', 'HML']
    available_cols = [c for c in factor_cols if c in ff3_factors.columns]

    results = run_ols_newey_west(
        excess_returns, ff3_factors[available_cols], lags=lags, add_constant=True
    )
    alpha_info = extract_alpha_tstats(results)

    output = {
        'model': 'FF3',
        'alpha': alpha_info['alpha'],
        'alpha_tstat': alpha_info['tstat'],
        'alpha_pvalue': alpha_info['pvalue'],
        'rsquared': results.rsquared,
        'nobs': int(results.nobs)
    }

    for col in available_cols:
        output[f'beta_{col}'] = results.params.get(col, np.nan)

    return output


def run_fama_french_5factor_model(
    excess_returns: pd.Series,
    ff5_factors: pd.DataFrame,
    lags: int = 4
) -> Dict:
    """
    Run Fama-French 5-factor regression: + RMW and CMA factors.

    Args:
        excess_returns: Portfolio excess returns
        ff5_factors: DataFrame with Mkt-RF, SMB, HML, RMW, CMA columns
        lags: Newey-West lags

    Returns:
        Dictionary with alpha, t-stat, factor loadings, and R²

    Examples:
        >>> excess_ret = pd.Series(np.random.normal(0.005, 0.05, 100))
        >>> ff5 = pd.DataFrame({
        ...     'Mkt-RF': np.random.normal(0.006, 0.04, 100),
        ...     'SMB': np.random.normal(0, 0.03, 100),
        ...     'HML': np.random.normal(0, 0.03, 100),
        ...     'RMW': np.random.normal(0, 0.02, 100),
        ...     'CMA': np.random.normal(0, 0.02, 100)
        ... })
        >>> result = run_fama_french_5factor_model(excess_ret, ff5)
        >>> 'alpha' in result
        True
    """
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    available_cols = [c for c in factor_cols if c in ff5_factors.columns]

    results = run_ols_newey_west(
        excess_returns, ff5_factors[available_cols], lags=lags, add_constant=True
    )
    alpha_info = extract_alpha_tstats(results)

    output = {
        'model': 'FF5',
        'alpha': alpha_info['alpha'],
        'alpha_tstat': alpha_info['tstat'],
        'alpha_pvalue': alpha_info['pvalue'],
        'rsquared': results.rsquared,
        'nobs': int(results.nobs)
    }

    for col in available_cols:
        output[f'beta_{col}'] = results.params.get(col, np.nan)

    return output


def test_all_models(
    excess_returns_df: pd.DataFrame,
    ff3_factors: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    lags: int = 4
) -> pd.DataFrame:
    """
    Test all factor portfolios against CAPM, FF3, and FF5 models.

    Args:
        excess_returns_df: DataFrame with factor excess returns (each column = 1 factor)
        ff3_factors: Fama-French 3-factor data
        ff5_factors: Fama-French 5-factor data
        lags: Newey-West lags

    Returns:
        DataFrame with results for all factors and all models

    Examples:
        >>> n = 100
        >>> factors = pd.DataFrame({
        ...     'momentum': np.random.normal(0.005, 0.05, n),
        ...     'value': np.random.normal(0.003, 0.04, n)
        ... })
        >>> ff3 = pd.DataFrame({
        ...     'Mkt-RF': np.random.normal(0.006, 0.04, n),
        ...     'SMB': np.random.normal(0, 0.03, n),
        ...     'HML': np.random.normal(0, 0.03, n)
        ... })
        >>> ff5 = ff3.copy()
        >>> ff5['RMW'] = np.random.normal(0, 0.02, n)
        >>> ff5['CMA'] = np.random.normal(0, 0.02, n)
        >>> results = test_all_models(factors, ff3, ff5)
        >>> 'factor' in results.columns
        True
    """
    all_results = []

    for factor_name in excess_returns_df.columns:
        factor_returns = excess_returns_df[factor_name].dropna()

        # Align with factors
        common_idx = factor_returns.index.intersection(ff3_factors.index)
        factor_aligned = factor_returns.reindex(common_idx)
        ff3_aligned = ff3_factors.reindex(common_idx)
        ff5_aligned = ff5_factors.reindex(common_idx)

        if len(factor_aligned) < 24:
            continue

        # CAPM
        capm_result = run_capm(
            factor_aligned, ff3_aligned['Mkt-RF'], lags=lags
        )
        capm_result['factor'] = factor_name
        all_results.append(capm_result)

        # FF3
        ff3_result = run_fama_french_3factor_model(
            factor_aligned, ff3_aligned, lags=lags
        )
        ff3_result['factor'] = factor_name
        all_results.append(ff3_result)

        # FF5
        ff5_result = run_fama_french_5factor_model(
            factor_aligned, ff5_aligned, lags=lags
        )
        ff5_result['factor'] = factor_name
        all_results.append(ff5_result)

    return pd.DataFrame(all_results)


def filter_significant_alphas(
    results_df: pd.DataFrame,
    significance_level: float = 0.05,
    require_all_models: bool = True
) -> pd.DataFrame:
    """
    Filter factors with statistically significant alphas.

    Args:
        results_df: Output from test_all_models
        significance_level: p-value threshold
        require_all_models: Require significance across CAPM, FF3, and FF5

    Returns:
        DataFrame with factors meeting significance criterion

    Examples:
        >>> results = pd.DataFrame({
        ...     'factor': ['mom', 'mom', 'mom', 'val', 'val', 'val'],
        ...     'model': ['CAPM', 'FF3', 'FF5', 'CAPM', 'FF3', 'FF5'],
        ...     'alpha_pvalue': [0.01, 0.02, 0.03, 0.1, 0.2, 0.3]
        ... })
        >>> sig = filter_significant_alphas(results, require_all_models=True)
        >>> 'mom' in sig['factor'].values
        True
    """
    # Mark significance
    results_df = results_df.copy()
    results_df['significant'] = results_df['alpha_pvalue'] < significance_level

    if require_all_models:
        # Count how many models show significance
        sig_count = results_df.groupby('factor')['significant'].sum()
        n_models = results_df.groupby('factor')['model'].count()

        # Factors significant in ALL models
        robust_factors = sig_count[sig_count == n_models].index.tolist()

        return results_df[results_df['factor'].isin(robust_factors)]
    else:
        return results_df[results_df['significant']]


def calculate_alpha_survival(
    results_df: pd.DataFrame,
    significance_level: float = 0.05
) -> pd.DataFrame:
    """
    Analyze alpha survival across increasingly stringent factor models.

    Tracks which factors maintain significant alphas as we move from CAPM
    to FF3 to FF5, showing which anomalies are robust to factor controls.

    Args:
        results_df: Output from test_all_models
        significance_level: p-value threshold

    Returns:
        DataFrame showing survival across models for each factor

    Examples:
        >>> results = pd.DataFrame({
        ...     'factor': ['mom', 'mom', 'mom'],
        ...     'model': ['CAPM', 'FF3', 'FF5'],
        ...     'alpha_pvalue': [0.01, 0.02, 0.04],
        ...     'alpha': [0.005, 0.004, 0.003]
        ... })
        >>> survival = calculate_alpha_survival(results)
        >>> 'CAPM_significant' in survival.columns
        True
    """
    results_df = results_df.copy()
    results_df['significant'] = results_df['alpha_pvalue'] < significance_level

    # Pivot to wide format
    survival = results_df.pivot_table(
        index='factor',
        columns='model',
        values=['alpha', 'alpha_tstat', 'significant']
    )

    # Flatten multi-level columns
    survival.columns = [f'{col[1]}_{col[0]}' for col in survival.columns]
    survival = survival.reset_index()

    # Add survival indicators
    models = ['CAPM', 'FF3', 'FF5']
    for model in models:
        sig_col = f'{model}_significant'
        if sig_col in survival.columns:
            survival[f'{model}_significant'] = survival[sig_col].fillna(False)

    return survival


def compute_benchmark_adjusted_returns(
    returns_df: pd.DataFrame,
    ff5_factors: pd.DataFrame,
    lags: int = 8
) -> pd.DataFrame:
    """
    Compute benchmark-adjusted (residual) returns using FF5 model.

    Args:
        returns_df: Factor returns DataFrame
        ff5_factors: Fama-French 5-factor data
        lags: Newey-West lags for residual calculation

    Returns:
        DataFrame with benchmark-adjusted returns

    Examples:
        >>> n = 100
        >>> returns = pd.DataFrame({'momentum': np.random.normal(0.005, 0.05, n)})
        >>> ff5 = pd.DataFrame({
        ...     'Mkt-RF': np.random.normal(0.006, 0.04, n),
        ...     'SMB': np.random.normal(0, 0.03, n),
        ...     'HML': np.random.normal(0, 0.03, n),
        ...     'RMW': np.random.normal(0, 0.02, n),
        ...     'CMA': np.random.normal(0, 0.02, n)
        ... })
        >>> residuals = compute_benchmark_adjusted_returns(returns, ff5)
        >>> residuals.shape[1] == returns.shape[1]
        True
    """
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    available_cols = [c for c in factor_cols if c in ff5_factors.columns]

    residuals_dict = {}

    for factor_name in returns_df.columns:
        try:
            # Align data
            combined = returns_df[[factor_name]].join(
                ff5_factors[available_cols], how='inner'
            ).dropna()

            if len(combined) < 24:
                continue

            # Calculate residuals
            residuals = get_residuals(
                combined,
                y_col=factor_name,
                factor_cols=available_cols,
                lags=lags
            )

            residuals_dict[factor_name] = residuals

        except Exception as e:
            print(f"Error computing residuals for {factor_name}: {e}")
            continue

    return pd.DataFrame(residuals_dict)
