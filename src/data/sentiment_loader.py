"""
Sentiment Indicator Data Loader

This module provides functions for loading all seven sentiment indicators
used in the analysis: BW, Zhou Investor, ICS, AAII, VIX, Manager, and Employee.
"""

import pandas as pd
import numpy as np
from typing import Optional
import os


def load_bw_sentiment(filepath: str, sheet_name: str = 'BW_EOM') -> pd.Series:
    """
    Load Baker-Wurgler Sentiment Index.

    Loads end-of-month composite sentiment index constructed from six
    market-based proxies: closed-end fund discount, NYSE share turnover,
    number of IPOs, first-day IPO returns, equity share in new issues,
    and dividend premium.

    Args:
        filepath: Path to BW_Sentiment.xlsx
        sheet_name: Sheet name ('BW_EOM' for end-of-month, 'BW_EOY' for end-of-year)

    Returns:
        Monthly sentiment series with datetime index

    Examples:
        >>> # df = load_bw_sentiment('data/raw/BW_Sentiment.xlsx')
        >>> pass
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Convert date columns to datetime index
    if 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.drop(columns=['year', 'month'], errors='ignore')
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

    # Extract sentiment column (typically named 'sent' or 'sentiment')
    sent_col = next((c for c in df.columns if 'sent' in c.lower()), df.columns[0])

    # Resample to month-end
    return df[sent_col].resample('M').last().rename('BW')


def load_zhou_investor_sentiment(filepath: str) -> pd.Series:
    """
    Load Zhou et al. Investor Sentiment Index.

    Args:
        filepath: Path to Zhou_InvestorSentiment.xlsx

    Returns:
        Monthly investor sentiment series

    Examples:
        >>> # series = load_zhou_investor_sentiment('data/raw/Zhou_InvestorSentiment.xlsx')
        >>> pass
    """
    df = pd.read_excel(filepath)

    if 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))
        df = df.drop(columns=['year', 'month'], errors='ignore')
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

    sent_col = next((c for c in df.columns if 'sent' in c.lower()), df.columns[0])
    return df[sent_col].resample('M').last().rename('Zhou_Investor')


def load_umich_consumer_sentiment(filepath: str) -> pd.DataFrame:
    """
    Load University of Michigan Consumer Sentiment (ICS, ICC, ICE).

    Args:
        filepath: Path to UniMichigan_ConsumerSentiment.csv

    Returns:
        DataFrame with ICS, ICC (conditions), and ICE (expectations) columns

    Examples:
        >>> # df = load_umich_consumer_sentiment('data/raw/UniMichigan_ConsumerSentiment.csv')
        >>> pass
    """
    df = pd.read_csv(filepath)

    # Convert date
    if 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    elif 'DATE' in df.columns:
        df.index = pd.to_datetime(df['DATE'])
        df = df.drop(columns=['DATE'])

    # Rename columns to standard names
    rename_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if 'ICS' in col_upper or 'INDEX' in col_upper:
            rename_map[col] = 'ICS'
        elif 'ICC' in col_upper or 'CURRENT' in col_upper:
            rename_map[col] = 'ICC'
        elif 'ICE' in col_upper or 'EXPECT' in col_upper:
            rename_map[col] = 'ICE'

    if rename_map:
        df = df.rename(columns=rename_map)

    return df.resample('M').last()


def load_aaii_sentiment(filepath: str) -> pd.Series:
    """
    Load AAII Sentiment Survey - Bull-Bear Spread.

    Args:
        filepath: Path to AAII_sentiment.xlsx

    Returns:
        Monthly Bull-Bear spread series

    Examples:
        >>> # series = load_aaii_sentiment('data/raw/AAII_sentiment.xlsx')
        >>> pass
    """
    df = pd.read_excel(filepath)

    # Convert date
    if 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    elif 'DATE' in df.columns:
        df.index = pd.to_datetime(df['DATE'])

    # Find Bull-Bear spread column
    bull_col = next((c for c in df.columns if 'bull' in c.lower()), None)
    bear_col = next((c for c in df.columns if 'bear' in c.lower()), None)
    spread_col = next((c for c in df.columns if 'spread' in c.lower()), None)

    if spread_col:
        series = df[spread_col]
    elif bull_col and bear_col:
        series = df[bull_col] - df[bear_col]
    else:
        series = df.iloc[:, 0]

    return series.resample('M').last().rename('AAII')


def load_vix(filepath: str) -> pd.Series:
    """
    Load CBOE Volatility Index (VIX).

    Note: VIX is an inverse sentiment proxy — high VIX indicates fear/low sentiment.

    Args:
        filepath: Path to CBoe_VIX.csv

    Returns:
        Monthly VIX series (end-of-month)

    Examples:
        >>> # series = load_vix('data/raw/CBoe_VIX.csv')
        >>> pass
    """
    df = pd.read_csv(filepath)

    # Common FRED format
    if 'DATE' in df.columns:
        df.index = pd.to_datetime(df['DATE'])
        df = df.drop(columns=['DATE'])
    elif 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

    # Replace missing values (FRED uses '.' for missing)
    df = df.replace('.', np.nan).astype(float)

    vix_col = df.columns[0]
    return df[vix_col].resample('M').last().rename('VIX')


def load_zhou_manager_sentiment(filepath: str) -> pd.Series:
    """
    Load Zhou et al. Manager Sentiment Index.

    Args:
        filepath: Path to Zhou_ManagerSentiment.xlsx

    Returns:
        Monthly manager sentiment series

    Examples:
        >>> # series = load_zhou_manager_sentiment('data/raw/Zhou_ManagerSentiment.xlsx')
        >>> pass
    """
    df = pd.read_excel(filepath)

    if 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    elif 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))

    sent_col = next((c for c in df.columns if 'sent' in c.lower()), df.columns[0])
    return df[sent_col].resample('M').last().rename('Manager')


def load_zhou_employee_sentiment(filepath: str) -> pd.Series:
    """
    Load Zhou et al. Employee Sentiment Index.

    Args:
        filepath: Path to Zhou_EmployeeSentiment.xlsx

    Returns:
        Monthly employee sentiment series

    Examples:
        >>> # series = load_zhou_employee_sentiment('data/raw/Zhou_EmployeeSentiment.xlsx')
        >>> pass
    """
    df = pd.read_excel(filepath)

    if 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    elif 'year' in df.columns and 'month' in df.columns:
        df.index = pd.to_datetime(df[['year', 'month']].assign(day=1))

    sent_col = next((c for c in df.columns if 'sent' in c.lower()), df.columns[0])
    return df[sent_col].resample('M').last().rename('Employee')


def load_all_sentiment_indicators(data_dir: str) -> pd.DataFrame:
    """
    Load all sentiment indicators from the data directory.

    Attempts to load all seven indicators. Missing files are skipped with a warning.

    Args:
        data_dir: Directory containing sentiment data files

    Returns:
        DataFrame with all available sentiment indicators

    Examples:
        >>> # df = load_all_sentiment_indicators('data/raw/sentiment')
        >>> pass
    """
    sentiment_series = []

    loaders = [
        ('BW_Sentiment.xlsx', 'Baker-Wurgler', load_bw_sentiment),
        ('Zhou_InvestorSentiment.xlsx', 'Zhou Investor', load_zhou_investor_sentiment),
        ('Zhou_ManagerSentiment.xlsx', 'Zhou Manager', load_zhou_manager_sentiment),
        ('Zhou_EmployeeSentiment.xlsx', 'Zhou Employee', load_zhou_employee_sentiment),
        ('CBoe_VIX.csv', 'VIX', load_vix),
    ]

    for filename, name, loader in loaders:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                series = loader(filepath)
                sentiment_series.append(series)
                print(f"Loaded: {name} ({len(series)} observations)")
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        else:
            print(f"Warning: {name} file not found: {filepath}")

    # University of Michigan (special case)
    umich_path = os.path.join(data_dir, 'UniMichigan_ConsumerSentiment.csv')
    if os.path.exists(umich_path):
        try:
            umich_df = load_umich_consumer_sentiment(umich_path)
            for col in umich_df.columns:
                sentiment_series.append(umich_df[col])
            print(f"Loaded: University of Michigan ({len(umich_df)} observations)")
        except Exception as e:
            print(f"Warning: Could not load University of Michigan: {e}")

    if not sentiment_series:
        raise FileNotFoundError(
            f"No sentiment files found in {data_dir}. "
            "See data/README.md for information on obtaining the data."
        )

    # Combine into DataFrame (outer join to preserve all dates)
    sentiment_df = pd.concat(sentiment_series, axis=1, join='outer')

    print(f"\nSentiment DataFrame: {sentiment_df.shape}")
    print(f"Date range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")

    return sentiment_df
