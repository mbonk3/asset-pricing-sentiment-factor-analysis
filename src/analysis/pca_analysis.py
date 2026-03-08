"""
Principal Component Analysis Module

This module provides PCA-based tools for dimensionality reduction
and common factor extraction from multiple sentiment indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    sentiment_df: pd.DataFrame,
    n_components: Optional[int] = None,
    standardize: bool = True
) -> Tuple[PCA, StandardScaler, np.ndarray]:
    """
    Run Principal Component Analysis on sentiment indicators.

    Args:
        sentiment_df: DataFrame with sentiment indicators as columns
        n_components: Number of components to retain (None = all)
        standardize: Whether to standardize before PCA

    Returns:
        Tuple of (pca_model, scaler, transformed_data)

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100),
        ...     'ICS': np.random.normal(85, 10, 100)
        ... })
        >>> pca, scaler, scores = run_pca(sentiment)
        >>> scores.shape[1] == 3
        True
    """
    # Remove missing values
    data = sentiment_df.dropna()

    if len(data) == 0:
        raise ValueError("No data available after removing missing values")

    # Standardize if requested
    scaler = StandardScaler()
    if standardize:
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data.values
        scaler.fit(data)  # Fit scaler even if not applying

    # Run PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_scaled)

    return pca, scaler, scores


def extract_loadings(
    pca_model: PCA,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract PCA component loadings.

    Args:
        pca_model: Fitted PCA model
        feature_names: Names of the original features

    Returns:
        DataFrame with loadings (rows = features, columns = components)

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100)
        ... })
        >>> pca, scaler, scores = run_pca(sentiment)
        >>> loadings = extract_loadings(pca, sentiment.columns.tolist())
        >>> loadings.shape[0] == len(sentiment.columns)
        True
    """
    n_components = pca_model.n_components_

    # Create component names
    component_names = [f'PC{i+1}' for i in range(n_components)]

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca_model.components_.T,
        index=feature_names,
        columns=component_names
    )

    return loadings


def calculate_explained_variance(
    pca_model: PCA
) -> pd.DataFrame:
    """
    Calculate explained variance for each component.

    Args:
        pca_model: Fitted PCA model

    Returns:
        DataFrame with variance explained per component

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100),
        ...     'ICS': np.random.normal(85, 10, 100)
        ... })
        >>> pca, scaler, scores = run_pca(sentiment)
        >>> var_explained = calculate_explained_variance(pca)
        >>> 'Explained_Variance_Ratio' in var_explained.columns
        True
    """
    n_components = pca_model.n_components_

    explained_var = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Eigenvalue': pca_model.explained_variance_,
        'Explained_Variance_Ratio': pca_model.explained_variance_ratio_,
        'Cumulative_Variance': np.cumsum(pca_model.explained_variance_ratio_)
    })

    return explained_var


def compute_principal_components(
    sentiment_df: pd.DataFrame,
    pca_model: PCA,
    scaler: StandardScaler
) -> pd.DataFrame:
    """
    Compute principal component scores for new data.

    Args:
        sentiment_df: DataFrame with sentiment indicators
        pca_model: Fitted PCA model
        scaler: Fitted StandardScaler

    Returns:
        DataFrame with PC scores

    Examples:
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, 100),
        ...     'VIX': np.random.normal(20, 5, 100)
        ... })
        >>> pca, scaler, _ = run_pca(sentiment)
        >>> scores = compute_principal_components(sentiment, pca, scaler)
        >>> scores.shape == (len(sentiment.dropna()), pca.n_components_)
        True
    """
    data = sentiment_df.dropna()

    # Scale
    data_scaled = scaler.transform(data)

    # Transform
    scores = pca_model.transform(data_scaled)

    # Create DataFrame
    component_names = [f'PC{i+1}' for i in range(pca_model.n_components_)]
    scores_df = pd.DataFrame(scores, index=data.index, columns=component_names)

    return scores_df


def pca_multiple_periods(
    sentiment_df: pd.DataFrame,
    periods: Dict[str, Tuple[str, str]],
    n_components: Optional[int] = None,
    standardize: bool = True
) -> Dict[str, Dict]:
    """
    Run PCA across multiple time periods for comparison.

    Args:
        sentiment_df: Full sentiment DataFrame
        periods: Dictionary mapping period names to (start_date, end_date) tuples
        n_components: Number of PCA components
        standardize: Whether to standardize

    Returns:
        Dictionary with PCA results for each period

    Examples:
        >>> dates = pd.date_range('1990-01', '2023-12', freq='M')
        >>> sentiment = pd.DataFrame({
        ...     'BW': np.random.normal(0, 1, len(dates)),
        ...     'VIX': np.random.normal(20, 5, len(dates))
        ... }, index=dates)
        >>> periods = {
        ...     'Full Sample': ('1990-01', '2023-12'),
        ...     'Post-2000': ('2000-01', '2023-12')
        ... }
        >>> results = pca_multiple_periods(sentiment, periods)
        >>> 'Full Sample' in results
        True
    """
    results = {}

    for period_name, (start, end) in periods.items():
        print(f"\nRunning PCA for period: {period_name} ({start} to {end})")

        # Filter to period
        period_data = sentiment_df.loc[start:end].dropna()

        if len(period_data) < 10:
            print(f"  Insufficient data: {len(period_data)} observations")
            results[period_name] = None
            continue

        print(f"  N observations: {len(period_data)}")
        print(f"  N indicators: {period_data.shape[1]}")

        try:
            # Run PCA
            pca, scaler, scores = run_pca(period_data, n_components, standardize)

            # Extract results
            loadings = extract_loadings(pca, period_data.columns.tolist())
            explained_var = calculate_explained_variance(pca)
            scores_df = compute_principal_components(period_data, pca, scaler)

            results[period_name] = {
                'pca_model': pca,
                'scaler': scaler,
                'loadings': loadings,
                'explained_variance': explained_var,
                'scores': scores_df,
                'n_obs': len(period_data),
                'n_indicators': period_data.shape[1]
            }

            # Print summary
            top_pc = explained_var.iloc[0]
            print(f"  PC1 explains: {top_pc['Explained_Variance_Ratio']:.1%}")
            print(f"  PC1+PC2 explains: {explained_var.iloc[:2]['Explained_Variance_Ratio'].sum():.1%}")

        except Exception as e:
            print(f"  Error: {e}")
            results[period_name] = None

    return results


def print_pca_summary(pca_results: Dict[str, Dict]) -> None:
    """
    Print summary of PCA results across periods.

    Args:
        pca_results: Output from pca_multiple_periods

    Examples:
        >>> results = {
        ...     'Full Sample': {
        ...         'explained_variance': pd.DataFrame({
        ...             'Component': ['PC1', 'PC2'],
        ...             'Explained_Variance_Ratio': [0.4, 0.25],
        ...             'Cumulative_Variance': [0.4, 0.65]
        ...         }),
        ...         'n_obs': 100,
        ...         'n_indicators': 5
        ...     }
        ... }
        >>> print_pca_summary(results)
        <BLANKLINE>
        PCA Summary
        ===========
        ...
    """
    print("\nPCA Summary")
    print("=" * 40)

    for period_name, result in pca_results.items():
        print(f"\n{period_name}:")

        if result is None:
            print("  No results available")
            continue

        print(f"  Observations: {result['n_obs']}")
        print(f"  Indicators: {result['n_indicators']}")

        # Explained variance
        ev = result['explained_variance']
        print("\n  Explained Variance:")
        for _, row in ev.iterrows():
            print(f"    {row['Component']}: {row['Explained_Variance_Ratio']:.1%} "
                  f"(cumulative: {row['Cumulative_Variance']:.1%})")

        # Top loadings for PC1
        loadings = result['loadings']
        if 'PC1' in loadings.columns:
            print("\n  PC1 Loadings:")
            pc1_sorted = loadings['PC1'].abs().sort_values(ascending=False)
            for indicator, loading in pc1_sorted.items():
                sign = '+' if loadings.loc[indicator, 'PC1'] > 0 else '-'
                print(f"    {indicator}: {sign}{abs(loadings.loc[indicator, 'PC1']):.3f}")
