"""
Sentiment-Conditional Analysis Module

This module provides functions for analyzing factor returns conditional on
sentiment regimes using tercile-based classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.statistical_utils import newey_west_tstat, calculate_terciles


def create_sentiment_terciles(
    sentiment_series: pd.Series,
    labels: List[str] = ['Low', 'Mid', 'High']
) -> pd.Series:
    """
    Create sentiment terciles (High/Mid/Low regimes).

    Splits sentiment into three groups based on percentiles. Typically,
    "Low" is bottom 33%, "Mid" is middle 33%, "High" is top 33%.

    Args:
        sentiment_series: Sentiment indicator
        labels: Labels for terciles (default: ['Low', 'Mid', 'High'])

    Returns:
        Categorical series with tercile labels

    Examples:
        >>> sentiment = pd.Series(np.random.normal(0, 1, 100))
        >>> terciles = create_sentiment_terciles(sentiment)
        >>> terciles.value_counts()['Low'] <= 34
        True
    """
    return calculate_terciles(sentiment_series, labels=labels)


def calculate_regime_returns(
    returns_df: pd.DataFrame,
    sentiment_terciles: pd.Series,
    regime_labels: List[str] = ['Low', 'High']
) -> pd.DataFrame:
    """
    Calculate mean returns conditional on sentiment regimes.

    Computes average returns for each factor during High and Low sentiment periods.

    Args:
        returns_df: DataFrame with factor returns (columns = factors, index = dates)
        sentiment_terciles: Series with tercile labels aligned to returns_df index
        regime_labels: Which regimes to analyze (default: ['Low', 'High'])

    Returns:
        DataFrame with mean returns for each regime

    Examples:
        >>> returns = pd.DataFrame({
        ...     'momentum': np.random.normal(0.05, 0.1, 100),
        ...     'value': np.random.normal(0.03, 0.1, 100)
        ... })
        >>> sentiment = pd.Series(np.random.normal(0, 1, 100), index=returns.index)
        >>> terciles = create_sentiment_terciles(sentiment)
        >>> regime_returns = calculate_regime_returns(returns, terciles)
        >>> 'Low' in regime_returns.index
        True
    """
    # Align returns and sentiment
    combined = returns_df.join(sentiment_terciles.rename('regime'), how='inner')

    results = {}

    for regime in regime_labels:
        # Filter to this regime
        regime_data = combined[combined['regime'] == regime]

        # Calculate mean returns for each factor
        mean_returns = regime_data.drop(columns=['regime']).mean()

        results[regime] = mean_returns

    # Convert to DataFrame
    regime_returns_df = pd.DataFrame(results).T

    return regime_returns_df


def compute_high_minus_low(
    returns_df: pd.DataFrame,
    sentiment_terciles: pd.Series,
    lags: int = 3
) -> pd.DataFrame:
    """
    Compute High-Minus-Low returns with Newey-West t-statistics.

    Calculates the difference in returns between High and Low sentiment regimes
    and tests for statistical significance using Newey-West t-statistics.

    Args:
        returns_df: DataFrame with factor returns
        sentiment_terciles: Series with tercile labels
        lags: Number of lags for Newey-West t-statistic

    Returns:
        DataFrame with High returns, Low returns, HML difference, and t-statistics

    Examples:
        >>> returns = pd.DataFrame({
        ...     'momentum': np.random.normal(0.05, 0.1, 100)
        ... })
        >>> sentiment = pd.Series(np.random.normal(0, 1, 100), index=returns.index)
        >>> terciles = create_sentiment_terciles(sentiment)
        >>> hml_df = compute_high_minus_low(returns, terciles)
        >>> 'HML' in hml_df.columns
        True
    """
    # Get regime returns
    regime_returns = calculate_regime_returns(
        returns_df,
        sentiment_terciles,
        regime_labels=['Low', 'High']
    )

    # Align data
    combined = returns_df.join(sentiment_terciles.rename('regime'), how='inner')

    results = []

    for factor in returns_df.columns:
        # High and Low mean returns
        high_mean = regime_returns.loc['High', factor]
        low_mean = regime_returns.loc['Low', factor]

        # HML difference
        hml = high_mean - low_mean

        # Get time series for HML t-stat calculation
        high_returns = combined[combined['regime'] == 'High'][factor]
        low_returns = combined[combined['regime'] == 'Low'][factor]

        # For HML t-stat, we need to construct the time series difference
        # Since High and Low are different periods, we can't directly subtract
        # Instead, we compute t-stat on the pooled data with a sign flip

        # Alternative: use the HML at each period where we have both
        # For now, let's compute separate means and combine

        # T-statistic for the difference
        # Use two-sample approach: t = (mean1 - mean2) / SE(mean1 - mean2)
        # Or use Newey-West on the regime-assigned series

        # Assign +1 for High, -1 for Low, 0 for Mid
        regime_signed = sentiment_terciles.map({'High': 1, 'Low': -1, 'Mid': 0})

        # Interaction term: returns * regime_signed
        interaction = returns_df[factor] * regime_signed

        # Remove Mid regime (where regime_signed = 0)
        interaction_filtered = interaction[regime_signed != 0]

        # Newey-West t-stat on interaction term
        if len(interaction_filtered) > lags + 1:
            t_stat = newey_west_tstat(interaction_filtered, lags=lags)
        else:
            t_stat = np.nan

        results.append({
            'Factor': factor,
            'High': high_mean,
            'Low': low_mean,
            'HML': hml,
            'HML_tstat': t_stat
        })

    hml_df = pd.DataFrame(results)

    return hml_df


def sentiment_factor_analysis(
    sentiment_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    portfolio_legs: List[str] = ['Long', 'Short', 'Long-Short'],
    lags: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Perform complete sentiment-conditional analysis.

    Analyzes how factor returns vary across sentiment regimes for each
    sentiment indicator and portfolio leg.

    Args:
        sentiment_df: DataFrame with sentiment indicators (columns = indicators)
        residuals_df: DataFrame with benchmark-adjusted returns (columns = factor_leg combinations)
        portfolio_legs: Which portfolio legs to analyze
        lags: Lags for Newey-West t-statistics

    Returns:
        Dictionary with results for each sentiment indicator and portfolio leg

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW_orth': np.random.normal(0, 1, 100)
        ... })
        >>> returns = pd.DataFrame({
        ...     'momentum_Long-Short': np.random.normal(0.05, 0.1, 100)
        ... })
        >>> sentiment.index = returns.index
        >>> results = sentiment_factor_analysis(sentiment, returns, ['Long-Short'])
        >>> isinstance(results, dict)
        True
    """
    all_results = {}

    for sent_indicator in sentiment_df.columns:
        print(f"\nAnalyzing sentiment indicator: {sent_indicator}")

        # Create terciles for this sentiment indicator
        terciles = create_sentiment_terciles(sentiment_df[sent_indicator])

        for leg in portfolio_legs:
            print(f"  Portfolio leg: {leg}")

            # Filter columns for this leg
            leg_columns = [col for col in residuals_df.columns if col.endswith(f'_{leg}')]

            if not leg_columns:
                print(f"    No columns found for {leg}")
                continue

            # Get returns for this leg
            leg_returns = residuals_df[leg_columns]

            # Rename columns to remove leg suffix for cleaner output
            leg_returns_clean = leg_returns.copy()
            leg_returns_clean.columns = [col.replace(f'_{leg}', '') for col in leg_columns]

            # Compute HML
            hml_results = compute_high_minus_low(leg_returns_clean, terciles, lags=lags)

            # Store results
            key = f"{sent_indicator}_{leg}"
            all_results[key] = hml_results

            # Print summary
            significant = hml_results[abs(hml_results['HML_tstat']) > 1.96]
            print(f"    Factors analyzed: {len(hml_results)}")
            print(f"    Significant (|t| > 1.96): {len(significant)}")

    return all_results


def aggregate_anomaly_returns(
    results_df: pd.DataFrame,
    group_cols: List[str] = ['date', 'sentiment', 'regime']
) -> pd.DataFrame:
    """
    Aggregate sentiment-conditional returns across all anomalies.

    Computes average returns across multiple factors for each date,
    sentiment indicator, and regime combination.

    Args:
        results_df: Long-format DataFrame with individual factor results
        group_cols: Columns to group by for aggregation

    Returns:
        Aggregated DataFrame

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': ['2020-01'] * 4,
        ...     'sentiment': ['BW'] * 4,
        ...     'regime': ['High', 'High', 'Low', 'Low'],
        ...     'factor': ['mom', 'value', 'mom', 'value'],
        ...     'return': [0.05, 0.03, 0.01, 0.02]
        ... })
        >>> agg = aggregate_anomaly_returns(df, ['sentiment', 'regime'])
        >>> 'return' in agg.columns
        True
    """
    # Group and aggregate
    agg_df = results_df.groupby(group_cols).agg({
        'return': 'mean',
        'factor': 'count'  # Count number of factors
    }).rename(columns={'factor': 'n_factors'}).reset_index()

    return agg_df


def create_sentiment_conditional_table(
    hml_results: pd.DataFrame,
    sort_by: str = 'HML_tstat',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Format sentiment-conditional results into publication-ready table.

    Args:
        hml_results: DataFrame from compute_high_minus_low
        sort_by: Column to sort by
        ascending: Sort order

    Returns:
        Formatted DataFrame

    Examples:
        >>> hml = pd.DataFrame({
        ...     'Factor': ['momentum', 'value'],
        ...     'High': [0.06, 0.04],
        ...     'Low': [0.02, 0.01],
        ...     'HML': [0.04, 0.03],
        ...     'HML_tstat': [2.5, 1.8]
        ... })
        >>> table = create_sentiment_conditional_table(hml)
        >>> table.columns[0] == 'Factor'
        True
    """
    # Sort
    formatted = hml_results.sort_values(by=sort_by, ascending=ascending).copy()

    # Round for display
    for col in ['High', 'Low', 'HML']:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(4)

    if 'HML_tstat' in formatted.columns:
        formatted['HML_tstat'] = formatted['HML_tstat'].round(2)

    # Add significance stars
    if 'HML_tstat' in formatted.columns:
        def add_stars(row):
            t = abs(row['HML_tstat'])
            if t > 2.576:  # 1% level
                return '***'
            elif t > 1.96:  # 5% level
                return '**'
            elif t > 1.645:  # 10% level
                return '*'
            else:
                return ''

        formatted['Significance'] = formatted.apply(add_stars, axis=1)

    return formatted


def pivot_results_for_heatmap(
    all_results: Dict[str, pd.DataFrame],
    value_col: str = 'HML_tstat'
) -> pd.DataFrame:
    """
    Pivot results for visualization as heatmap.

    Args:
        all_results: Dictionary from sentiment_factor_analysis
        value_col: Which value to use for heatmap ('HML', 'HML_tstat', etc.)

    Returns:
        Pivoted DataFrame suitable for heatmap

    Examples:
        >>> results = {
        ...     'BW_Long-Short': pd.DataFrame({
        ...         'Factor': ['momentum', 'value'],
        ...         'HML_tstat': [2.5, 1.8]
        ...     })
        ... }
        >>> pivoted = pivot_results_for_heatmap(results)
        >>> isinstance(pivoted, pd.DataFrame)
        True
    """
    # Combine all results
    combined_list = []

    for key, df in all_results.items():
        # Parse key
        parts = key.rsplit('_', 1)
        if len(parts) == 2:
            sentiment_ind, portfolio_leg = parts
        else:
            sentiment_ind = key
            portfolio_leg = 'Unknown'

        # Add metadata
        df_copy = df.copy()
        df_copy['Sentiment'] = sentiment_ind
        df_copy['Portfolio_Leg'] = portfolio_leg

        combined_list.append(df_copy)

    if not combined_list:
        return pd.DataFrame()

    # Combine all
    combined = pd.concat(combined_list, ignore_index=True)

    # Pivot
    pivoted = combined.pivot_table(
        index='Factor',
        columns=['Sentiment', 'Portfolio_Leg'],
        values=value_col
    )

    return pivoted
