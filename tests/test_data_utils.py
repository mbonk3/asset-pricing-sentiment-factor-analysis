"""
Tests for data utility functions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_utils import (
    compute_returns,
    clean_dataset,
    flip_negative_ls_returns,
    calculate_excess_returns,
    pivot_for_analysis
)
from utils.date_utils import (
    to_datetime_index,
    find_continuous_start_end_sentiment,
    resample_to_month_end,
    calculate_growth_rate
)


class TestToDatetimeIndex:
    """Tests for datetime index conversion."""

    def test_year_month_columns(self):
        df = pd.DataFrame({
            'year': [2020, 2020, 2020],
            'month': [1, 2, 3],
            'value': [10, 20, 30]
        })
        result = to_datetime_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert 'year' not in result.columns
        assert 'month' not in result.columns

    def test_date_column(self):
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
            'value': [10, 20, 30]
        })
        result = to_datetime_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_already_datetime_index(self):
        dates = pd.date_range('2020-01', periods=3, freq='M')
        df = pd.DataFrame({'value': [10, 20, 30]}, index=dates)
        result = to_datetime_index(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 3

    def test_raises_on_unconvertible(self):
        df = pd.DataFrame({'value': [1, 2, 3]})  # No date info
        with pytest.raises(ValueError):
            to_datetime_index(df)

    def test_preserves_data_values(self):
        df = pd.DataFrame({
            'year': [2020, 2020],
            'month': [1, 2],
            'value': [42, 99]
        })
        result = to_datetime_index(df)
        assert result['value'].iloc[0] == 42
        assert result['value'].iloc[1] == 99


class TestFindContinuousStartEnd:
    """Tests for continuous period identification."""

    def test_full_continuous_period(self):
        dates = pd.date_range('2020-01', periods=12, freq='M')
        df = pd.DataFrame({'value': range(12)}, index=dates)
        start, end = find_continuous_start_end_sentiment(df, 'value')
        assert start == dates[0]
        assert end == dates[-1]

    def test_period_with_missing_values(self):
        dates = pd.date_range('2020-01', periods=10, freq='M')
        values = [1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, 10]
        df = pd.DataFrame({'value': values}, index=dates)
        start, end = find_continuous_start_end_sentiment(df, 'value')
        # Should find one of the continuous segments
        assert start is not None
        assert end is not None

    def test_all_missing_returns_none(self):
        dates = pd.date_range('2020-01', periods=5, freq='M')
        df = pd.DataFrame({'value': [np.nan] * 5}, index=dates)
        start, end = find_continuous_start_end_sentiment(df, 'value')
        assert start is None
        assert end is None

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['value'])
        start, end = find_continuous_start_end_sentiment(df, 'value')
        assert start is None
        assert end is None


class TestComputeReturns:
    """Tests for Long/Short/Long-Short return computation."""

    def test_basic_computation(self):
        row = pd.Series({'01': 0.02, '10': 0.08})
        result = compute_returns(row)
        assert result['Long'] == 0.08
        assert result['Short'] == 0.02
        assert abs(result['Long-Short'] - 0.06) < 1e-10

    def test_missing_decile_gives_nan(self):
        row = pd.Series({'01': 0.02})  # Missing '10'
        result = compute_returns(row)
        assert np.isnan(result['Long'])
        assert np.isnan(result['Long-Short'])

    def test_negative_long_short(self):
        row = pd.Series({'01': 0.08, '10': 0.02})
        result = compute_returns(row)
        assert result['Long-Short'] < 0


class TestFlipNegativeLSReturns:
    """Tests for Long-Short return sign flipping."""

    def test_flips_negative_mean(self):
        df = pd.DataFrame({
            'signal': ['momentum'] * 5,
            'Long-Short': [-0.01, -0.02, -0.015, -0.01, -0.005]
        })
        result = flip_negative_ls_returns(df)
        assert result['Long-Short'].mean() > 0

    def test_does_not_flip_positive_mean(self):
        df = pd.DataFrame({
            'signal': ['value'] * 5,
            'Long-Short': [0.01, 0.02, 0.015, 0.01, 0.005]
        })
        result = flip_negative_ls_returns(df)
        assert result['Long-Short'].mean() > 0

    def test_does_not_modify_original(self):
        df = pd.DataFrame({
            'signal': ['momentum'] * 3,
            'Long-Short': [-0.01, -0.02, -0.015]
        })
        original_values = df['Long-Short'].copy()
        flip_negative_ls_returns(df)
        pd.testing.assert_series_equal(df['Long-Short'], original_values)

    def test_handles_multiple_signals(self):
        df = pd.DataFrame({
            'signal': ['momentum', 'momentum', 'value', 'value'],
            'Long-Short': [-0.01, -0.02, 0.01, 0.02]
        })
        result = flip_negative_ls_returns(df)
        # Both should have positive mean
        for sig in ['momentum', 'value']:
            group_mean = result[result['signal'] == sig]['Long-Short'].mean()
            assert group_mean > 0


class TestCalculateExcessReturns:
    """Tests for excess return computation."""

    def test_subtracts_rf_correctly(self):
        df = pd.DataFrame({
            'ret': [0.05, 0.06, 0.04],
            'RF': [0.01, 0.01, 0.01]
        })
        result = calculate_excess_returns(df, rf_col='RF', return_cols=['ret'])
        expected = [0.04, 0.05, 0.03]
        for actual, exp in zip(result['ret'], expected):
            assert abs(actual - exp) < 1e-10

    def test_raises_without_rf_column(self):
        df = pd.DataFrame({'ret': [0.05, 0.06]})
        with pytest.raises(ValueError):
            calculate_excess_returns(df, rf_col='RF', return_cols=['ret'])

    def test_does_not_subtract_rf_from_itself(self):
        df = pd.DataFrame({
            'ret': [0.05, 0.06],
            'RF': [0.01, 0.01]
        })
        result = calculate_excess_returns(df, rf_col='RF', return_cols=['ret'])
        # RF column should remain unchanged
        assert list(result['RF']) == [0.01, 0.01]


class TestCalculateGrowthRate:
    """Tests for growth rate computation."""

    def test_log_growth_rate(self):
        series = pd.Series([100.0, 102.0, 104.04])
        growth = calculate_growth_rate(series, periods=1, method='log')
        # log(102/100) ≈ 0.0198
        assert abs(growth.iloc[1] - np.log(102 / 100)) < 1e-10

    def test_pct_growth_rate(self):
        series = pd.Series([100.0, 105.0, 110.25])
        growth = calculate_growth_rate(series, periods=1, method='pct')
        assert abs(growth.iloc[1] - 0.05) < 1e-10

    def test_12_month_growth(self):
        series = pd.Series([100.0] * 12 + [110.0])
        growth = calculate_growth_rate(series, periods=12, method='pct')
        assert abs(growth.iloc[-1] - 0.10) < 1e-10

    def test_invalid_method_raises(self):
        series = pd.Series([100.0, 105.0])
        with pytest.raises(ValueError):
            calculate_growth_rate(series, method='invalid')
