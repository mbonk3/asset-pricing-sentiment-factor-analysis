"""
Tests for orthogonalization functions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.orthogonalization import (
    transform_to_growth_rates,
    orthogonalize_sentiment,
    orthogonalize_all_indicators
)


class TestTransformToGrowthRates:
    """Tests for macroeconomic growth rate transformation."""

    def setup_method(self):
        self.n = 36  # 36 periods
        self.dates = pd.date_range('2000-01', periods=self.n, freq='M')
        self.df = pd.DataFrame({
            'INDPRO': 100 * (1.02 ** (np.arange(self.n) / 12)),
            'CPI': 100 * (1.03 ** (np.arange(self.n) / 12))
        }, index=self.dates)

    def test_returns_dataframe(self):
        result = transform_to_growth_rates(
            self.df, real_vars=['INDPRO'], nominal_vars=['CPI'], periods=12
        )
        assert isinstance(result, pd.DataFrame)

    def test_creates_growth_columns(self):
        result = transform_to_growth_rates(
            self.df, real_vars=['INDPRO'], nominal_vars=['CPI'], periods=12
        )
        assert 'INDPRO_growth' in result.columns
        assert 'CPI_growth' in result.columns

    def test_shorter_than_original(self):
        """Growth rates lose observations due to lag."""
        result = transform_to_growth_rates(
            self.df, real_vars=['INDPRO'], nominal_vars=[], periods=12
        )
        assert len(result) < len(self.df)

    def test_log_growth_approximately_correct(self):
        """Annual log-growth of 2% should give ~0.02 per period."""
        result = transform_to_growth_rates(
            self.df, real_vars=['INDPRO'], nominal_vars=[], periods=12
        )
        # log(1.02) ≈ 0.0198
        mean_growth = result['INDPRO_growth'].mean()
        assert abs(mean_growth - np.log(1.02)) < 0.01

    def test_pct_growth_approximately_correct(self):
        """Annual pct-growth of 3% should give ~0.03 per period."""
        result = transform_to_growth_rates(
            self.df, real_vars=[], nominal_vars=['CPI'], periods=12
        )
        mean_growth = result['CPI_growth'].mean()
        assert abs(mean_growth - 0.03) < 0.005

    def test_missing_variable_skipped_gracefully(self):
        """Variables not in df should be skipped without error."""
        result = transform_to_growth_rates(
            self.df,
            real_vars=['INDPRO', 'NON_EXISTENT'],
            nominal_vars=[],
            periods=12
        )
        assert 'INDPRO_growth' in result.columns
        assert 'NON_EXISTENT_growth' not in result.columns


class TestOrthogonalizeSentiment:
    """Tests for sentiment orthogonalization."""

    def setup_method(self):
        np.random.seed(42)
        self.n = 100
        self.dates = pd.date_range('2000-01', periods=self.n, freq='M')

        # Create macro variables
        self.macro_df = pd.DataFrame({
            'INDPRO_growth': np.random.normal(0.02, 0.05, self.n),
            'CPI_growth': np.random.normal(0.03, 0.02, self.n)
        }, index=self.dates)

        # Create sentiment with macro component
        self.sentiment = pd.Series(
            0.5 * self.macro_df['INDPRO_growth'] + np.random.normal(0, 0.5, self.n),
            index=self.dates
        )

    def test_returns_tuple(self):
        result, diag = orthogonalize_sentiment(self.sentiment, self.macro_df)
        assert isinstance(result, pd.Series)
        assert isinstance(diag, dict)

    def test_residuals_have_low_macro_correlation(self):
        """Orthogonalized sentiment should have lower correlation with macro than original."""
        orthogonalized, _ = orthogonalize_sentiment(
            self.sentiment, self.macro_df, run_diagnostics_flag=False
        )

        # Original correlation
        orig_corr = abs(self.sentiment.corr(self.macro_df['INDPRO_growth']))

        # Orthogonalized correlation (should be near zero)
        orth_data = pd.DataFrame({
            'orth': orthogonalized,
            'macro': self.macro_df['INDPRO_growth']
        }).dropna()
        orth_corr = abs(orth_data['orth'].corr(orth_data['macro']))

        assert orth_corr < orig_corr
        assert orth_corr < 0.1  # Should be very small

    def test_diagnostics_contain_rsquared(self):
        _, diagnostics = orthogonalize_sentiment(self.sentiment, self.macro_df)
        assert 'rsquared' in diagnostics

    def test_empty_result_on_empty_overlap(self):
        """No overlap between sentiment and macro should return empty."""
        future_dates = pd.date_range('2050-01', periods=10, freq='M')
        future_sentiment = pd.Series(np.random.normal(0, 1, 10), index=future_dates)

        result, diag = orthogonalize_sentiment(future_sentiment, self.macro_df)
        assert len(result) == 0


class TestOrthogonalizeAllIndicators:
    """Tests for batch orthogonalization."""

    def setup_method(self):
        np.random.seed(42)
        self.n = 100
        self.dates = pd.date_range('2000-01', periods=self.n, freq='M')

        self.macro_df = pd.DataFrame({
            'INDPRO_growth': np.random.normal(0.02, 0.05, self.n),
            'CPI_growth': np.random.normal(0.03, 0.02, self.n)
        }, index=self.dates)

        self.sentiment_df = pd.DataFrame({
            'BW': np.random.normal(0, 1, self.n),
            'VIX': np.random.normal(20, 5, self.n),
            'ICS': np.random.normal(85, 10, self.n)
        }, index=self.dates)

    def test_returns_dataframe_and_dict(self):
        result_df, diagnostics = orthogonalize_all_indicators(
            self.sentiment_df, self.macro_df, run_diagnostics_flag=False
        )
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(diagnostics, dict)

    def test_correct_number_of_output_columns(self):
        """Should produce one orthogonalized column per input column."""
        result_df, _ = orthogonalize_all_indicators(
            self.sentiment_df, self.macro_df, run_diagnostics_flag=False
        )
        assert len(result_df.columns) == len(self.sentiment_df.columns)

    def test_output_columns_have_orth_suffix(self):
        result_df, _ = orthogonalize_all_indicators(
            self.sentiment_df, self.macro_df, run_diagnostics_flag=False
        )
        for col in result_df.columns:
            assert col.endswith('_orth')

    def test_diagnostics_for_each_indicator(self):
        _, diagnostics = orthogonalize_all_indicators(
            self.sentiment_df, self.macro_df, run_diagnostics_flag=True
        )
        for indicator in self.sentiment_df.columns:
            assert indicator in diagnostics
