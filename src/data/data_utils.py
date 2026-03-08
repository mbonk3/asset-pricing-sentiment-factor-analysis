"""
Data Utility Functions

This module provides utilities for data loading, cleaning, and processing
factor portfolios and sentiment indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import os
import yaml


def load_config(config_path: str = 'config/config.yaml') -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary with configuration parameters

    Examples:
        >>> config = load_config()
        >>> 'paths' in config
        True
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_returns(row: pd.Series) -> pd.Series:
    """
    Calculate Long, Short, and Long-Short returns from decile data.

    Takes a row containing decile portfolio returns (01 through 10) and
    computes returns for long portfolio (decile 10), short portfolio (decile 01),
    and long-short portfolio (10 - 01).

    Args:
        row: Series with decile returns (columns: '01', '02', ..., '10')

    Returns:
        Series with 'Long', 'Short', and 'Long-Short' returns

    Examples:
        >>> row = pd.Series({f'{i:02d}': i * 0.01 for i in range(1, 11)})
        >>> returns = compute_returns(row)
        >>> 'Long-Short' in returns
        True
    """
    long = row.get('10', np.nan)
    short = row.get('01', np.nan)
    long_short = long - short if (pd.notna(long) and pd.notna(short)) else np.nan

    return pd.Series({
        'Long': long,
        'Short': short,
        'Long-Short': long_short
    })


def clean_dataset(
    df: pd.DataFrame,
    source_type: str,
    is_region: bool,
    mapping_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Clean and standardize factor/theme datasets with mapping.

    Applies consistent naming conventions, filters, and transformations
    to factor portfolio data from different sources (OAP, JKP).

    Args:
        df: Raw factor portfolio DataFrame
        source_type: 'factor' or 'theme'
        is_region: Whether data is regional (True) or country-level (False)
        mapping_df: Optional DataFrame with factor name mappings

    Returns:
        Cleaned and standardized DataFrame

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01', periods=5, freq='M'),
        ...     'signal': ['momentum'] * 5,
        ...     'ret': np.random.normal(0, 0.1, 5)
        ... })
        >>> cleaned = clean_dataset(df, 'factor', False)
        >>> 'date' in cleaned.columns
        True
    """
    df = df.copy()

    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Apply mapping if provided
    if mapping_df is not None and 'signal' in df.columns:
        # Merge with mapping to get standardized names
        if source_type == 'factor':
            map_col = 'factor_name'
        else:
            map_col = 'theme_name'

        if map_col in mapping_df.columns:
            df = df.merge(
                mapping_df[['signal', map_col]],
                on='signal',
                how='left'
            )
            # Use mapped name if available
            df['signal'] = df[map_col].fillna(df['signal'])
            df = df.drop(columns=[map_col])

    # Add source metadata
    df['source_type'] = source_type
    df['is_region'] = is_region

    # Remove duplicates
    subset_cols = ['date', 'signal']
    if 'country' in df.columns:
        subset_cols.append('country')

    df = df.drop_duplicates(subset=subset_cols, keep='last')

    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date')

    return df


def flip_negative_ls_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Long-Short portfolios have positive mean returns.

    Some factor portfolios may be defined in reverse (e.g., low minus high).
    This function flips the signs if the mean Long-Short return is negative,
    ensuring consistent interpretation.

    Args:
        df: DataFrame with Long-Short returns

    Returns:
        DataFrame with flipped returns where appropriate

    Examples:
        >>> df = pd.DataFrame({
        ...     'signal': ['momentum'] * 3,
        ...     'Long-Short': [-0.01, -0.02, -0.015]
        ... })
        >>> flipped = flip_negative_ls_returns(df)
        >>> flipped['Long-Short'].mean() > 0
        True
    """
    df = df.copy()

    if 'Long-Short' not in df.columns:
        return df

    if 'signal' in df.columns:
        # Flip by signal
        for signal in df['signal'].unique():
            mask = df['signal'] == signal
            mean_ls = df.loc[mask, 'Long-Short'].mean()

            if mean_ls < 0:
                # Flip all returns for this signal
                df.loc[mask, 'Long-Short'] = -df.loc[mask, 'Long-Short']

                if 'Long' in df.columns:
                    df.loc[mask, 'Long'] = -df.loc[mask, 'Long']

                if 'Short' in df.columns:
                    df.loc[mask, 'Short'] = -df.loc[mask, 'Short']
    else:
        # Flip entire DataFrame
        mean_ls = df['Long-Short'].mean()
        if mean_ls < 0:
            df['Long-Short'] = -df['Long-Short']

            if 'Long' in df.columns:
                df['Long'] = -df['Long']

            if 'Short' in df.columns:
                df['Short'] = -df['Short']

    return df


def save_checkpoint(
    df: pd.DataFrame,
    name: str,
    path: str = 'results/checkpoints',
    encoding: str = 'utf-8-sig'
) -> None:
    """
    Save intermediate results as checkpoint.

    Args:
        df: DataFrame to save
        name: Checkpoint name (without extension)
        path: Directory to save checkpoint
        encoding: File encoding

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> save_checkpoint(df, 'test_checkpoint', path='.')
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f'{name}.csv')
    df.to_csv(filepath, encoding=encoding, index=True)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    name: str,
    path: str = 'results/checkpoints',
    encoding: str = 'utf-8-sig',
    index_col: int = 0
) -> pd.DataFrame:
    """
    Load checkpoint from file.

    Args:
        name: Checkpoint name (without extension)
        path: Directory containing checkpoint
        encoding: File encoding
        index_col: Column to use as index

    Returns:
        DataFrame loaded from checkpoint

    Examples:
        >>> df = load_checkpoint('test_checkpoint', path='.')
        >>> isinstance(df, pd.DataFrame)
        True
    """
    filepath = os.path.join(path, f'{name}.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    df = pd.read_csv(filepath, encoding=encoding, index_col=index_col)
    print(f"Checkpoint loaded: {filepath}")

    return df


def calculate_excess_returns(
    df: pd.DataFrame,
    rf_col: str = 'RF',
    return_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate excess returns by subtracting risk-free rate.

    Args:
        df: DataFrame with returns and risk-free rate
        rf_col: Name of risk-free rate column
        return_cols: List of return columns to adjust (if None, adjusts all numeric columns except RF)

    Returns:
        DataFrame with excess returns

    Examples:
        >>> df = pd.DataFrame({
        ...     'ret': [0.05, 0.06, 0.04],
        ...     'RF': [0.01, 0.01, 0.01]
        ... })
        >>> excess = calculate_excess_returns(df, return_cols=['ret'])
        >>> excess['ret'].mean() < df['ret'].mean()
        True
    """
    df = df.copy()

    if rf_col not in df.columns:
        raise ValueError(f"Risk-free rate column '{rf_col}' not found")

    # Determine which columns to adjust
    if return_cols is None:
        # Adjust all numeric columns except RF
        return_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if rf_col in return_cols:
            return_cols.remove(rf_col)

    # Subtract risk-free rate
    for col in return_cols:
        if col in df.columns:
            df[col] = df[col] - df[rf_col]

    return df


def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    on: List[str],
    how: str = 'outer',
    suffixes: Tuple[str, str] = ('_oap', '_jkp')
) -> pd.DataFrame:
    """
    Merge two datasets with consistent naming.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        on: Columns to merge on
        how: Merge type ('inner', 'outer', 'left', 'right')
        suffixes: Suffixes for overlapping columns

    Returns:
        Merged DataFrame

    Examples:
        >>> df1 = pd.DataFrame({'date': ['2020-01'], 'signal': ['momentum'], 'ret1': [0.05]})
        >>> df2 = pd.DataFrame({'date': ['2020-01'], 'signal': ['momentum'], 'ret2': [0.06]})
        >>> merged = merge_datasets(df1, df2, on=['date', 'signal'])
        >>> 'ret1' in merged.columns and 'ret2' in merged.columns
        True
    """
    merged = df1.merge(df2, on=on, how=how, suffixes=suffixes)
    return merged


def pivot_for_analysis(
    df: pd.DataFrame,
    index: str = 'date',
    columns: str = 'signal',
    values: str = 'ret'
) -> pd.DataFrame:
    """
    Pivot DataFrame for time series analysis.

    Transforms long-format data to wide-format with dates as index
    and signals as columns.

    Args:
        df: Long-format DataFrame
        index: Column to use as index (typically 'date')
        columns: Column to use as columns (typically 'signal')
        values: Column to use as values (typically returns)

    Returns:
        Pivoted DataFrame

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': ['2020-01', '2020-01', '2020-02', '2020-02'],
        ...     'signal': ['mom', 'value', 'mom', 'value'],
        ...     'ret': [0.05, 0.03, 0.04, 0.06]
        ... })
        >>> pivoted = pivot_for_analysis(df)
        >>> 'mom' in pivoted.columns
        True
    """
    pivoted = df.pivot(index=index, columns=columns, values=values)
    return pivoted


def filter_continuous_periods(
    df: pd.DataFrame,
    min_periods: int = 36
) -> pd.DataFrame:
    """
    Filter out signals with insufficient continuous data.

    Args:
        df: DataFrame with signals
        min_periods: Minimum number of continuous periods required

    Returns:
        Filtered DataFrame

    Examples:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01', periods=40, freq='M'),
        ...     'signal': ['momentum'] * 40,
        ...     'ret': np.random.normal(0, 0.1, 40)
        ... })
        >>> filtered = filter_continuous_periods(df, min_periods=36)
        >>> len(filtered) >= 36
        True
    """
    df = df.copy()

    if 'signal' in df.columns:
        # Filter by signal
        valid_signals = []

        for signal, group in df.groupby('signal'):
            if len(group) >= min_periods:
                valid_signals.append(signal)

        df = df[df['signal'].isin(valid_signals)]

    else:
        # Filter entire DataFrame
        if len(df) < min_periods:
            df = pd.DataFrame()  # Empty

    return df
