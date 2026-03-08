"""
Statistical Tests Module

This module provides statistical testing functions including
correlation analysis, FDR correction, and alpha survival tests.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.statistical_utils import pearson_correlation_test, fdr_correction


def correlation_analysis(
    sentiment_df: pd.DataFrame,
    periods: Optional[Dict[str, Tuple[str, str]]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Compute pairwise correlations across sentiment indicators and time periods.

    Args:
        sentiment_df: DataFrame with sentiment indicators
        periods: Optional dict of {period_name: (start, end)} for sub-period analysis

    Returns:
        Dictionary mapping period names to correlation matrices

    Examples:
        >>> import numpy as np, pandas as pd
        >>> dates = pd.date_range('2000-01', periods=100, freq='M')
        >>> sentiment = pd.DataFrame({'BW': np.random.normal(0,1,100),
        ...                           'VIX': np.random.normal(20,5,100)}, index=dates)
        >>> results = correlation_analysis(sentiment)
        >>> 'Full Sample' in results
        True
    """
    results = {}

    # Always compute full sample
    full_corr = sentiment_df.corr(method='pearson')
    results['Full Sample'] = full_corr

    # Compute for additional periods
    if periods:
        for period_name, (start, end) in periods.items():
            period_data = sentiment_df.loc[start:end]
            if len(period_data) >= 10:
                results[period_name] = period_data.corr(method='pearson')

    return results


def compute_pairwise_pvalues(
    sentiment_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute pairwise p-values for Pearson correlations.

    Args:
        sentiment_df: DataFrame with sentiment indicators

    Returns:
        DataFrame with p-values for each pair

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100)
        ... })
        >>> pvalues = compute_pairwise_pvalues(sentiment)
        >>> pvalues.shape == (2, 2)
        True
    """
    cols = sentiment_df.columns.tolist()
    pvalue_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i == j:
                pvalue_matrix.loc[col1, col2] = 0.0
            elif i < j:
                result = pearson_correlation_test(
                    sentiment_df[col1], sentiment_df[col2]
                )
                pvalue_matrix.loc[col1, col2] = result['pvalue']
                pvalue_matrix.loc[col2, col1] = result['pvalue']

    return pvalue_matrix


def fdr_correction_correlations(
    pvalue_df: pd.DataFrame,
    alpha: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply FDR correction to correlation p-values.

    Args:
        pvalue_df: DataFrame with p-values
        alpha: FDR level

    Returns:
        Tuple of (corrected_pvalues_df, rejected_df)

    Examples:
        >>> pvalues = pd.DataFrame({'a': [0.001, 0.05], 'b': [0.05, 0.0]},
        ...                        index=['a', 'b'])
        >>> corrected, rejected = fdr_correction_correlations(pvalues)
        >>> corrected.shape == pvalues.shape
        True
    """
    # Flatten upper triangle only
    cols = pvalue_df.columns.tolist()
    flat_pvalues = []
    flat_pairs = []

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i < j:
                p = pvalue_df.loc[col1, col2]
                if not np.isnan(p):
                    flat_pvalues.append(p)
                    flat_pairs.append((col1, col2))

    if not flat_pvalues:
        return pvalue_df.copy(), pvalue_df.copy()

    # Apply FDR
    rejected, corrected = fdr_correction(flat_pvalues, alpha=alpha)

    # Reconstruct matrices
    corrected_df = pd.DataFrame(np.nan, index=cols, columns=cols)
    rejected_df = pd.DataFrame(False, index=cols, columns=cols)

    for (col1, col2), corr_p, rej in zip(flat_pairs, corrected, rejected):
        corrected_df.loc[col1, col2] = corr_p
        corrected_df.loc[col2, col1] = corr_p
        rejected_df.loc[col1, col2] = rej
        rejected_df.loc[col2, col1] = rej

    return corrected_df, rejected_df


def alpha_survival_test(
    alpha_results: pd.DataFrame,
    significance_levels: List[float] = [0.01, 0.05, 0.10]
) -> pd.DataFrame:
    """
    Test how many factors survive alpha tests at different significance levels.

    Args:
        alpha_results: DataFrame with 'factor', 'model', 'alpha_pvalue' columns
        significance_levels: List of significance thresholds to test

    Returns:
        Summary DataFrame with survival counts

    Examples:
        >>> results = pd.DataFrame({
        ...     'factor': ['mom', 'mom', 'mom', 'val', 'val', 'val'],
        ...     'model': ['CAPM', 'FF3', 'FF5'] * 2,
        ...     'alpha_pvalue': [0.01, 0.02, 0.03, 0.1, 0.2, 0.3]
        ... })
        >>> survival = alpha_survival_test(results)
        >>> 'model' in survival.columns
        True
    """
    summary_rows = []

    models = alpha_results['model'].unique()
    total_factors = alpha_results['factor'].nunique()

    for model in models:
        model_data = alpha_results[alpha_results['model'] == model]
        row = {'model': model, 'total': total_factors}

        for level in significance_levels:
            col_name = f'sig_{int(level*100)}pct'
            sig_count = (model_data['alpha_pvalue'] < level).sum()
            row[col_name] = sig_count
            row[f'{col_name}_pct'] = sig_count / total_factors * 100

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def print_correlation_table(
    title: str,
    corr_df: pd.DataFrame,
    pvalue_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Print a formatted correlation table.

    Args:
        title: Table title
        corr_df: Correlation matrix
        pvalue_df: Optional p-value matrix for significance marking

    Examples:
        >>> corr = pd.DataFrame({'a': [1.0, 0.5], 'b': [0.5, 1.0]},
        ...                     index=['a', 'b'])
        >>> print_correlation_table('Test', corr)
        <BLANKLINE>
        Test
        ...
    """
    print(f"\n{title}")
    print("=" * len(title))

    if pvalue_df is not None:
        # Format with significance stars
        formatted = corr_df.copy().astype(str)
        for col in corr_df.columns:
            for idx in corr_df.index:
                corr_val = corr_df.loc[idx, col]
                p_val = pvalue_df.loc[idx, col] if idx in pvalue_df.index else np.nan

                if np.isnan(corr_val):
                    formatted.loc[idx, col] = '-'
                else:
                    stars = ''
                    if not np.isnan(p_val):
                        if p_val < 0.01:
                            stars = '***'
                        elif p_val < 0.05:
                            stars = '**'
                        elif p_val < 0.10:
                            stars = '*'
                    formatted.loc[idx, col] = f'{corr_val:.3f}{stars}'

        print(formatted.to_string())
    else:
        print(corr_df.round(3).to_string())

    if pvalue_df is not None:
        print("\n* p<0.10, ** p<0.05, *** p<0.01")
