"""
Sentiment Orthogonalization Module

This module provides functions for orthogonalizing sentiment indicators
against macroeconomic variables to isolate the behavioral component.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.regression_utils import run_ols_newey_west, regression_summary_dict
from utils.statistical_utils import compute_vif, breusch_pagan_test


def load_macro_variables(
    filepath: str,
    sheet_name: str = 'BW_macro'
) -> pd.DataFrame:
    """
    Load Baker-Wurgler macroeconomic variables.

    Args:
        filepath: Path to Excel file with macro data
        sheet_name: Sheet name containing macro variables

    Returns:
        DataFrame with macroeconomic variables

    Examples:
        >>> # df = load_macro_variables('data/raw/BW_Sentiment.xlsx')
        >>> # 'INDPRO' in df.columns
        >>> pass
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Convert to datetime if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    elif 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.drop(columns=['year', 'month'], errors='ignore')

    return df


def transform_to_growth_rates(
    df: pd.DataFrame,
    real_vars: List[str],
    nominal_vars: List[str],
    periods: int = 12
) -> pd.DataFrame:
    """
    Transform macroeconomic variables to growth rates.

    Converts real variables using log-differences and nominal variables
    using percentage changes over the specified period.

    Args:
        df: DataFrame with macro variables
        real_vars: List of real variable names (use log-difference)
        nominal_vars: List of nominal variable names (use pct change)
        periods: Number of periods for growth calculation (default: 12 for annual)

    Returns:
        DataFrame with growth rates

    Examples:
        >>> df = pd.DataFrame({
        ...     'INDPRO': [100, 102, 104, 106, 108],
        ...     'CPI': [200, 202, 204, 206, 208]
        ... })
        >>> growth = transform_to_growth_rates(df, ['INDPRO'], ['CPI'], periods=1)
        >>> 'INDPRO_growth' in growth.columns
        True
    """
    growth_df = pd.DataFrame(index=df.index)

    # Log-differences for real variables
    for var in real_vars:
        if var in df.columns:
            growth_df[f'{var}_growth'] = np.log(df[var]) - np.log(df[var].shift(periods))

    # Percentage changes for nominal variables
    for var in nominal_vars:
        if var in df.columns:
            growth_df[f'{var}_growth'] = df[var].pct_change(periods=periods)

    # Drop missing values from growth calculations
    growth_df = growth_df.dropna()

    return growth_df


def orthogonalize_sentiment(
    sentiment_series: pd.Series,
    macro_df: pd.DataFrame,
    lags: int = 4,
    run_diagnostics_flag: bool = True
) -> Tuple[pd.Series, Dict]:
    """
    Orthogonalize sentiment indicator against macroeconomic variables.

    Regresses sentiment on macro growth rates using OLS with Newey-West HAC
    standard errors. Residuals represent the orthogonalized (behavioral)
    component of sentiment.

    Args:
        sentiment_series: Sentiment indicator to orthogonalize
        macro_df: DataFrame with macroeconomic growth rates
        lags: Number of lags for Newey-West adjustment
        run_diagnostics_flag: Whether to run diagnostic tests

    Returns:
        Tuple of (orthogonalized_sentiment, diagnostics_dict)

    Examples:
        >>> sentiment = pd.Series(np.random.normal(0, 1, 100))
        >>> macro = pd.DataFrame({
        ...     'INDPRO_growth': np.random.normal(0, 0.1, 100),
        ...     'CPI_growth': np.random.normal(0, 0.05, 100)
        ... })
        >>> orth_sent, diag = orthogonalize_sentiment(sentiment, macro, lags=4)
        >>> len(orth_sent) > 0
        True
    """
    # Align sentiment and macro data
    combined = pd.concat([sentiment_series, macro_df], axis=1, join='inner')
    combined = combined.dropna()

    if len(combined) == 0:
        return pd.Series(), {}

    # Separate y and X
    y = combined.iloc[:, 0]  # First column is sentiment
    X = combined.iloc[:, 1:]  # Rest are macro variables

    # Run OLS with Newey-West
    results = run_ols_newey_west(y, X, lags=lags, add_constant=True)

    # Orthogonalized sentiment = residuals
    orthogonalized = results.resid

    # Diagnostics
    diagnostics = {}

    if run_diagnostics_flag:
        # Basic regression stats
        diagnostics.update(regression_summary_dict(results))

        # VIF
        if len(X.columns) > 1:
            vif_df = compute_vif(X, X.columns.tolist())
            diagnostics['vif'] = vif_df.to_dict('records')

        # Breusch-Pagan test
        try:
            # Need to add constant to X for Breusch-Pagan
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X)
            bp_results = breusch_pagan_test(results.resid.values, X_with_const.values)
            diagnostics['breusch_pagan'] = bp_results
        except Exception as e:
            diagnostics['breusch_pagan'] = {'error': str(e)}

    return orthogonalized, diagnostics


def orthogonalize_all_indicators(
    sentiment_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    lags: int = 4,
    run_diagnostics_flag: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Orthogonalize all sentiment indicators against macro variables.

    Applies orthogonalization to each sentiment indicator in the DataFrame.

    Args:
        sentiment_df: DataFrame with sentiment indicators (each column is an indicator)
        macro_df: DataFrame with macroeconomic growth rates
        lags: Number of lags for Newey-West adjustment
        run_diagnostics_flag: Whether to run diagnostic tests

    Returns:
        Tuple of (orthogonalized_df, diagnostics_by_indicator)

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100)
        ... })
        >>> macro = pd.DataFrame({
        ...     'INDPRO_growth': np.random.normal(0, 0.1, 100),
        ...     'CPI_growth': np.random.normal(0, 0.05, 100)
        ... })
        >>> orth_df, diag = orthogonalize_all_indicators(sentiment, macro)
        >>> len(orth_df.columns) == len(sentiment.columns)
        True
    """
    orthogonalized_dict = {}
    diagnostics_dict = {}

    for col in sentiment_df.columns:
        print(f"\nOrthogonalizing {col}...")

        # Orthogonalize this indicator
        orth_series, diagnostics = orthogonalize_sentiment(
            sentiment_df[col],
            macro_df,
            lags=lags,
            run_diagnostics_flag=run_diagnostics_flag
        )

        # Store results
        orthogonalized_dict[f'{col}_orth'] = orth_series
        diagnostics_dict[col] = diagnostics

        # Print summary
        if diagnostics and 'rsquared' in diagnostics:
            print(f"  R-squared: {diagnostics['rsquared']:.4f}")
            print(f"  N observations: {diagnostics.get('nobs', 'N/A')}")

    # Combine into DataFrame
    orthogonalized_df = pd.DataFrame(orthogonalized_dict)

    return orthogonalized_df, diagnostics_dict


def print_orthogonalization_summary(
    diagnostics_dict: Dict
) -> None:
    """
    Print summary of orthogonalization results.

    Args:
        diagnostics_dict: Dictionary of diagnostics from orthogonalize_all_indicators

    Examples:
        >>> diagnostics = {
        ...     'BW': {'rsquared': 0.25, 'nobs': 100},
        ...     'VIX': {'rsquared': 0.15, 'nobs': 100}
        ... }
        >>> print_orthogonalization_summary(diagnostics)
        <BLANKLINE>
        Orthogonalization Summary
        =========================
        <BLANKLINE>
        Indicator: BW
          R-squared: 0.2500
          Observations: 100
        ...
    """
    print("\nOrthogonalization Summary")
    print("=" * 25)

    for indicator, diagnostics in diagnostics_dict.items():
        print(f"\nIndicator: {indicator}")

        if not diagnostics:
            print("  No diagnostics available")
            continue

        # Key statistics
        if 'rsquared' in diagnostics:
            print(f"  R-squared: {diagnostics['rsquared']:.4f}")

        if 'nobs' in diagnostics:
            print(f"  Observations: {diagnostics['nobs']}")

        # VIF
        if 'vif' in diagnostics:
            print("  VIF (Multicollinearity):")
            for vif_info in diagnostics['vif']:
                print(f"    {vif_info['Feature']}: {vif_info['VIF']:.2f}")

        # Breusch-Pagan
        if 'breusch_pagan' in diagnostics:
            bp = diagnostics['breusch_pagan']
            if 'lm_pvalue' in bp:
                print(f"  Breusch-Pagan p-value: {bp['lm_pvalue']:.4f}")


def combine_original_and_orthogonalized(
    original_df: pd.DataFrame,
    orthogonalized_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine original and orthogonalized sentiment indicators.

    Args:
        original_df: Original sentiment indicators
        orthogonalized_df: Orthogonalized sentiment indicators

    Returns:
        Combined DataFrame

    Examples:
        >>> original = pd.DataFrame({'BW': [1, 2, 3]})
        >>> orth = pd.DataFrame({'BW_orth': [0.5, 1.0, 1.5]})
        >>> combined = combine_original_and_orthogonalized(original, orth)
        >>> 'BW' in combined.columns and 'BW_orth' in combined.columns
        True
    """
    combined = pd.concat([original_df, orthogonalized_df], axis=1, join='outer')
    return combined
