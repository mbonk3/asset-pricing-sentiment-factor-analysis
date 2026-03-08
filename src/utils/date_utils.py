"""
Date and Time Utility Functions

This module provides utilities for handling date conversions, time series alignment,
and identifying continuous data periods in financial time series.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert month/year columns to datetime index.

    Handles various date column formats (year/month, date, etc.) and converts
    them to a proper datetime index for time series analysis.

    Args:
        df: DataFrame with date information in columns (e.g., 'year', 'month', or 'date')

    Returns:
        DataFrame with datetime index

    Examples:
        >>> df = pd.DataFrame({'year': [2020, 2020], 'month': [1, 2], 'value': [10, 20]})
        >>> df = to_datetime_index(df)
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    df = df.copy()

    # Check if already has datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # Try to construct datetime from year/month columns
    if 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.drop(columns=['year', 'month'], errors='ignore')

    # Try to use 'date' column
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

    # Try to convert index to datetime
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            raise ValueError(
                "Could not convert to datetime index. "
                "DataFrame must have 'year'/'month' columns, 'date' column, "
                "or convertible index."
            )

    return df


def find_continuous_start_end_sentiment(
    subdf: pd.DataFrame,
    value_col: str
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Identify continuous non-missing periods for sentiment data.

    Finds the start and end dates of the longest continuous period without
    missing values in a time series. Useful for determining valid date ranges
    for analysis.

    Args:
        subdf: DataFrame with datetime index
        value_col: Name of the column to check for continuity

    Returns:
        Tuple of (start_date, end_date) for the continuous period,
        or (None, None) if no valid period found

    Examples:
        >>> dates = pd.date_range('2020-01', periods=5, freq='M')
        >>> df = pd.DataFrame({'value': [1, 2, np.nan, 4, 5]}, index=dates)
        >>> start, end = find_continuous_start_end_sentiment(df, 'value')
        >>> start
        Timestamp('2020-01-31 00:00:00', freq='M')
    """
    if subdf.empty or value_col not in subdf.columns:
        return None, None

    # Find non-missing values
    non_missing = subdf[value_col].notna()

    if not non_missing.any():
        return None, None

    # Find first and last non-missing
    first_valid = non_missing.idxmax()  # First True
    last_valid = subdf[non_missing].index[-1]  # Last True

    # Check if continuous between first and last
    date_range = subdf.loc[first_valid:last_valid]

    if date_range[value_col].isna().any():
        # Has gaps - find longest continuous stretch
        # Create groups of continuous non-missing values
        groups = (non_missing != non_missing.shift()).cumsum()

        # Filter to only non-missing groups
        valid_groups = groups[non_missing]

        if valid_groups.empty:
            return None, None

        # Find longest group
        group_sizes = valid_groups.value_counts()
        longest_group = group_sizes.idxmax()

        # Get start and end of longest group
        longest_period = subdf[groups == longest_group]
        start_date = longest_period.index[0]
        end_date = longest_period.index[-1]
    else:
        # No gaps - use first and last valid
        start_date = first_valid
        end_date = last_valid

    return start_date, end_date


def find_continuous_start_end_reverse(
    subdf: pd.DataFrame
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Identify continuous periods from grouped data (reverse approach).

    Alternative method for finding continuous date ranges when data is already
    grouped or when you want to work backwards from the end.

    Args:
        subdf: DataFrame with datetime index

    Returns:
        Tuple of (start_date, end_date) for the continuous period,
        or (None, None) if no valid period found

    Examples:
        >>> dates = pd.date_range('2020-01', periods=5, freq='M')
        >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=dates)
        >>> start, end = find_continuous_start_end_reverse(df)
        >>> end
        Timestamp('2020-05-31 00:00:00', freq='M')
    """
    if subdf.empty:
        return None, None

    # Sort by date to ensure chronological order
    subdf = subdf.sort_index()

    # Get first and last dates
    start_date = subdf.index[0]
    end_date = subdf.index[-1]

    return start_date, end_date


def resample_to_month_end(
    df: pd.DataFrame,
    method: str = 'last'
) -> pd.DataFrame:
    """
    Resample time series to month-end frequency.

    Converts various time series frequencies (daily, weekly, etc.) to
    month-end frequency for alignment with other monthly financial data.

    Args:
        df: DataFrame with datetime index
        method: Resampling method - 'last' (last observation), 'mean', 'sum', etc.

    Returns:
        DataFrame resampled to month-end frequency

    Examples:
        >>> dates = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> df = pd.DataFrame({'value': range(10)}, index=dates)
        >>> monthly = resample_to_month_end(df, method='mean')
        >>> isinstance(monthly.index, pd.DatetimeIndex)
        True
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    # Already monthly - just ensure month-end
    if df.index.freq == 'M' or df.index.freq == 'ME':
        return df

    # Resample based on method
    if method == 'last':
        return df.resample('M').last()
    elif method == 'mean':
        return df.resample('M').mean()
    elif method == 'sum':
        return df.resample('M').sum()
    elif method == 'first':
        return df.resample('M').first()
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def align_time_series(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    join: str = 'inner'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two time series DataFrames by their datetime indices.

    Ensures both DataFrames have the same date range, useful for merging
    sentiment data with return data.

    Args:
        df1: First DataFrame with datetime index
        df2: Second DataFrame with datetime index
        join: Join type - 'inner', 'outer', 'left', or 'right'

    Returns:
        Tuple of (aligned_df1, aligned_df2) with matching indices

    Examples:
        >>> dates1 = pd.date_range('2020-01', periods=5, freq='M')
        >>> dates2 = pd.date_range('2020-03', periods=5, freq='M')
        >>> df1 = pd.DataFrame({'a': range(5)}, index=dates1)
        >>> df2 = pd.DataFrame({'b': range(5)}, index=dates2)
        >>> aligned1, aligned2 = align_time_series(df1, df2, join='inner')
        >>> len(aligned1) == len(aligned2)
        True
    """
    if not isinstance(df1.index, pd.DatetimeIndex):
        raise ValueError("df1 must have DatetimeIndex")
    if not isinstance(df2.index, pd.DatetimeIndex):
        raise ValueError("df2 must have DatetimeIndex")

    # Create combined index based on join type
    if join == 'inner':
        common_index = df1.index.intersection(df2.index)
    elif join == 'outer':
        common_index = df1.index.union(df2.index)
    elif join == 'left':
        common_index = df1.index
    elif join == 'right':
        common_index = df2.index
    else:
        raise ValueError(f"Unknown join type: {join}")

    # Reindex both DataFrames
    aligned_df1 = df1.reindex(common_index)
    aligned_df2 = df2.reindex(common_index)

    return aligned_df1, aligned_df2


def calculate_growth_rate(
    series: pd.Series,
    periods: int = 12,
    method: str = 'log'
) -> pd.Series:
    """
    Calculate growth rates for time series data.

    Computes either log-differences (for real variables) or percentage changes
    (for nominal variables) over a specified window.

    Args:
        series: Time series data
        periods: Number of periods for growth calculation (default: 12 for annual)
        method: 'log' for log-differences or 'pct' for percentage change

    Returns:
        Series with growth rates

    Examples:
        >>> series = pd.Series([100, 105, 110, 115, 120])
        >>> growth = calculate_growth_rate(series, periods=1, method='pct')
        >>> growth.iloc[1]
        0.05
    """
    if method == 'log':
        # Log-difference for real variables
        return np.log(series) - np.log(series.shift(periods))
    elif method == 'pct':
        # Percentage change for nominal variables
        return series.pct_change(periods=periods)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'log' or 'pct'")
