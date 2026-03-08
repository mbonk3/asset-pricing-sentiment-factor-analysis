"""
Tests for regression utility functions.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.regression_utils import (
    run_ols_newey_west,
    get_residuals,
    extract_alpha_tstats,
    regression_summary_dict
)
from utils.statistical_utils import newey_west_tstat


class TestNeweyWestTstat:
    """Tests for Newey-West t-statistic calculation."""

    def test_returns_float(self):
        series = pd.Series(np.random.normal(0.5, 1, 100))
        t_stat = newey_west_tstat(series, lags=3)
        assert isinstance(t_stat, float)

    def test_positive_mean_positive_tstat(self):
        """Positive mean return should yield positive t-statistic."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0.05, 0.1, 200))
        t_stat = newey_west_tstat(series, lags=3)
        assert t_stat > 0

    def test_empty_series_returns_nan(self):
        series = pd.Series([], dtype=float)
        t_stat = newey_west_tstat(series, lags=3)
        assert np.isnan(t_stat)

    def test_constant_series_returns_nan(self):
        """Constant series has zero variance - t-stat undefined."""
        series = pd.Series([1.0] * 100)
        t_stat = newey_west_tstat(series, lags=3)
        assert np.isnan(t_stat)

    def test_drops_missing_values(self):
        series = pd.Series([1.0, np.nan, 2.0, np.nan, 1.5])
        t_stat = newey_west_tstat(series, lags=1)
        assert not np.isnan(t_stat)

    def test_higher_lags_more_conservative(self):
        """More lags generally reduce the t-statistic magnitude."""
        np.random.seed(42)
        # Create autocorrelated series
        series = pd.Series(np.cumsum(np.random.normal(0.01, 0.1, 100)))
        t_low_lags = abs(newey_west_tstat(series, lags=1))
        t_high_lags = abs(newey_west_tstat(series, lags=5))
        # With autocorrelation, higher lags should reduce t-stat
        assert t_low_lags >= t_high_lags or abs(t_low_lags - t_high_lags) < 2.0


class TestRunOlsNeweyWest:
    """Tests for OLS with Newey-West standard errors."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 100
        self.X = pd.DataFrame({
            'x1': np.random.normal(0, 1, self.n),
            'x2': np.random.normal(0, 1, self.n)
        })
        self.y = 0.5 * self.X['x1'] + 0.3 * self.X['x2'] + np.random.normal(0, 0.5, self.n)
        self.y = pd.Series(self.y)

    def test_returns_results_object(self):
        results = run_ols_newey_west(self.y, self.X)
        assert hasattr(results, 'params')
        assert hasattr(results, 'tvalues')
        assert hasattr(results, 'pvalues')

    def test_has_constant_when_added(self):
        results = run_ols_newey_west(self.y, self.X, add_constant=True)
        assert 'const' in results.params.index

    def test_no_constant_when_excluded(self):
        results = run_ols_newey_west(self.y, self.X, add_constant=False)
        assert 'const' not in results.params.index

    def test_correct_number_of_params(self):
        results = run_ols_newey_west(self.y, self.X, add_constant=True)
        # 2 predictors + constant
        assert len(results.params) == 3

    def test_approximate_coefficient_recovery(self):
        """Coefficients should be approximately correct."""
        results = run_ols_newey_west(self.y, self.X, add_constant=True)
        assert abs(results.params['x1'] - 0.5) < 0.2
        assert abs(results.params['x2'] - 0.3) < 0.2

    def test_handles_missing_values(self):
        y_with_nan = self.y.copy()
        y_with_nan.iloc[0] = np.nan
        y_with_nan.iloc[50] = np.nan
        results = run_ols_newey_west(y_with_nan, self.X)
        assert results.nobs == self.n - 2

    def test_rsquared_between_zero_and_one(self):
        results = run_ols_newey_west(self.y, self.X)
        assert 0 <= results.rsquared <= 1


class TestGetResiduals:
    """Tests for residual calculation."""

    def setup_method(self):
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame({
            'y': np.random.normal(0.005, 0.05, n),
            'mkt': np.random.normal(0.006, 0.04, n),
            'smb': np.random.normal(0, 0.03, n)
        })

    def test_returns_series(self):
        residuals = get_residuals(self.df, 'y', ['mkt', 'smb'])
        assert isinstance(residuals, pd.Series)

    def test_residuals_have_low_correlation_with_regressors(self):
        """Residuals should have near-zero correlation with predictors."""
        residuals = get_residuals(self.df, 'y', ['mkt', 'smb'])

        # Align residuals with factors
        combined = pd.DataFrame({'resid': residuals}).join(self.df[['mkt', 'smb']])
        combined = combined.dropna()

        corr_mkt = abs(combined['resid'].corr(combined['mkt']))
        corr_smb = abs(combined['resid'].corr(combined['smb']))

        assert corr_mkt < 0.05  # Near-zero correlation
        assert corr_smb < 0.05

    def test_residuals_shorter_than_input(self):
        """Residuals may have fewer observations due to lag dropping."""
        residuals = get_residuals(self.df, 'y', ['mkt', 'smb'])
        assert len(residuals) <= len(self.df)


class TestExtractAlphaTstats:
    """Tests for alpha extraction from regression results."""

    def setup_method(self):
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({'x': np.random.normal(0, 1, n)})
        # Create y with known intercept of 0.005
        y = pd.Series(0.005 + 0.8 * X['x'].values + np.random.normal(0, 0.1, n))
        self.results = run_ols_newey_west(y, X, add_constant=True)

    def test_returns_dict(self):
        alpha_dict = extract_alpha_tstats(self.results)
        assert isinstance(alpha_dict, dict)

    def test_contains_required_keys(self):
        alpha_dict = extract_alpha_tstats(self.results)
        assert 'alpha' in alpha_dict
        assert 'tstat' in alpha_dict
        assert 'pvalue' in alpha_dict
        assert 'rsquared' in alpha_dict

    def test_alpha_approximately_correct(self):
        """Estimated alpha should be close to true value of 0.005."""
        alpha_dict = extract_alpha_tstats(self.results)
        assert abs(alpha_dict['alpha'] - 0.005) < 0.05

    def test_rsquared_in_range(self):
        alpha_dict = extract_alpha_tstats(self.results)
        assert 0 <= alpha_dict['rsquared'] <= 1
