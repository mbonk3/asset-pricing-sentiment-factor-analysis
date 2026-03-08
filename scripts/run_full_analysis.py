"""
Full Analysis Pipeline

This script runs the complete asset pricing and investor sentiment analysis,
from raw data loading through final result generation.

Usage:
    python scripts/run_full_analysis.py
    python scripts/run_full_analysis.py --config config/config.yaml
    python scripts/run_full_analysis.py --skip-orthogonalization
    python scripts/run_full_analysis.py --load-checkpoints
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_utils import load_config, save_checkpoint, load_checkpoint
from preprocessing.orthogonalization import (
    transform_to_growth_rates,
    orthogonalize_all_indicators,
    print_orthogonalization_summary
)
from analysis.factor_models import (
    test_all_models,
    filter_significant_alphas,
    calculate_alpha_survival,
    compute_benchmark_adjusted_returns
)
from analysis.sentiment_conditional import (
    sentiment_factor_analysis,
    create_sentiment_conditional_table,
    pivot_results_for_heatmap
)
from analysis.pca_analysis import (
    pca_multiple_periods,
    print_pca_summary
)
from analysis.statistical_tests import (
    correlation_analysis,
    alpha_survival_test
)
from visualization.time_series_plots import (
    plot_correlation_heatmap,
    plot_pca_loadings,
    plot_hml_tstatistics,
    plot_alpha_survival
)

warnings.filterwarnings('ignore')


def setup_logging(log_level: str, log_file: str) -> None:
    """Configure logging for the analysis pipeline."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Asset Pricing and Investor Sentiment Analysis Pipeline'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--skip-orthogonalization',
        action='store_true',
        help='Skip orthogonalization step (load from checkpoint)'
    )

    parser.add_argument(
        '--skip-factor-models',
        action='store_true',
        help='Skip factor model regressions (load from checkpoint)'
    )

    parser.add_argument(
        '--load-checkpoints',
        action='store_true',
        help='Load all data from checkpoints (skip processing)'
    )

    parser.add_argument(
        '--sentiment-indicators',
        nargs='+',
        default=None,
        help='Specific sentiment indicators to analyze (default: all)'
    )

    parser.add_argument(
        '--portfolio-legs',
        nargs='+',
        default=['Long', 'Short', 'Long-Short'],
        help='Portfolio legs to analyze'
    )

    return parser.parse_args()


def create_output_dirs(config: dict) -> None:
    """Create output directories."""
    for path_key in ['figures', 'tables', 'checkpoints']:
        os.makedirs(config['paths'][path_key], exist_ok=True)

    logging.info("Output directories created")


def load_sentiment_data(config: dict) -> pd.DataFrame:
    """
    Load all sentiment indicators.

    Returns:
        DataFrame with all sentiment indicators
    """
    logging.info("Loading sentiment indicators...")

    # In practice, load from actual files
    # Here we provide the structure for loading
    sentiment_path = config['paths']['raw_data']

    # Attempt to load from checkpoint first
    checkpoint_path = os.path.join(config['paths']['checkpoints'], 'sentiment_df.csv')

    if os.path.exists(checkpoint_path):
        logging.info("Loading sentiment from checkpoint...")
        df = pd.read_csv(checkpoint_path, index_col=0, parse_dates=True)
        return df

    # Otherwise load from raw files
    # This section should be expanded with actual file loading code
    logging.warning(
        "Raw sentiment files not found. See data/README.md for data sources. "
        "Using placeholder data for demonstration."
    )

    # Placeholder data structure matching your analysis
    dates = pd.date_range('1965-01-01', '2023-12-01', freq='M')
    df = pd.DataFrame({
        'BW': np.random.normal(0, 1, len(dates)),
        'Zhou_Investor': np.random.normal(0, 1, len(dates)),
        'ICS': np.random.normal(85, 10, len(dates)),
        'AAII': np.random.normal(0, 20, len(dates)),
        'VIX': np.abs(np.random.normal(20, 5, len(dates))),
        'Manager': np.random.normal(0, 1, len(dates)),
        'Employee': np.random.normal(0, 1, len(dates))
    }, index=dates)

    return df


def load_macro_data(config: dict) -> pd.DataFrame:
    """
    Load macroeconomic variables for orthogonalization.

    Returns:
        DataFrame with macro growth rates
    """
    logging.info("Loading macroeconomic data...")

    # Placeholder macro data
    dates = pd.date_range('1965-01-01', '2023-12-01', freq='M')
    macro_raw = pd.DataFrame({
        'INDPRO': 100 * (1.02 ** (np.arange(len(dates)) / 12)),
        'DURABLE': 100 * (1.015 ** (np.arange(len(dates)) / 12)),
        'NONDUR': 100 * (1.01 ** (np.arange(len(dates)) / 12)),
        'SERVICES': 100 * (1.025 ** (np.arange(len(dates)) / 12)),
        'EMPLOY': 100 * (1.005 ** (np.arange(len(dates)) / 12)),
        'CPI': 100 * (1.03 ** (np.arange(len(dates)) / 12))
    }, index=dates)

    # Transform to growth rates
    real_vars = config['macro_variables']['real_variables']
    nominal_vars = config['macro_variables']['nominal_variables']
    periods = config['macro_variables']['growth_window']

    macro_growth = transform_to_growth_rates(macro_raw, real_vars, nominal_vars, periods)

    return macro_growth


def run_orthogonalization(
    sentiment_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: dict,
    skip: bool = False
) -> pd.DataFrame:
    """Run sentiment orthogonalization step."""

    checkpoint_name = 'orthogonalized_sentiment'
    checkpoint_file = os.path.join(
        config['paths']['checkpoints'], f'{checkpoint_name}.csv'
    )

    if skip or os.path.exists(checkpoint_file):
        logging.info("Loading orthogonalized sentiment from checkpoint...")
        return load_checkpoint(checkpoint_name, config['paths']['checkpoints'])

    logging.info("Running sentiment orthogonalization...")

    lags = config['regression']['newey_west_lags']['orthogonalization']

    orthogonalized_df, diagnostics = orthogonalize_all_indicators(
        sentiment_df,
        macro_df,
        lags=lags,
        run_diagnostics_flag=True
    )

    print_orthogonalization_summary(diagnostics)

    # Save checkpoint
    save_checkpoint(orthogonalized_df, checkpoint_name, config['paths']['checkpoints'])

    return orthogonalized_df


def run_pca_analysis(
    orthogonalized_df: pd.DataFrame,
    config: dict
) -> dict:
    """Run PCA across multiple time periods."""
    logging.info("Running PCA analysis...")

    # Define sample periods
    periods = {
        'Full Sample': ('1965-01', '2023-12'),
        'Long Sample': ('1970-01', '2023-12'),
        'Modern Sample': ('1990-01', '2023-12'),
        'Post-2000': ('2000-01', '2023-12')
    }

    pca_results = pca_multiple_periods(orthogonalized_df, periods)
    print_pca_summary(pca_results)

    # Save loadings for each period
    for period_name, result in pca_results.items():
        if result is not None:
            period_safe = period_name.replace(' ', '_').lower()
            result['loadings'].to_csv(
                os.path.join(config['paths']['tables'],
                             f'pca_loadings_{period_safe}.csv')
            )
            result['explained_variance'].to_csv(
                os.path.join(config['paths']['tables'],
                             f'pca_variance_{period_safe}.csv')
            )

    return pca_results


def load_factor_data(config: dict) -> pd.DataFrame:
    """
    Load factor portfolio data.

    Returns:
        DataFrame with factor returns
    """
    logging.info("Loading factor portfolio data...")

    checkpoint_file = os.path.join(
        config['paths']['checkpoints'], 'factor_returns.csv'
    )

    if os.path.exists(checkpoint_file):
        return load_checkpoint('factor_returns', config['paths']['checkpoints'])

    logging.warning(
        "Factor data not found. See data/README.md for data sources. "
        "Using placeholder data for demonstration."
    )

    # Placeholder factor data
    dates = pd.date_range('1990-01-01', '2023-12-01', freq='M')
    factors = ['Momentum', 'Value', 'Size', 'Profitability', 'Investment',
               'Low_Vol', 'Quality', 'Growth', 'Reversal', 'Dividend']

    factor_returns = pd.DataFrame(index=dates)

    for factor in factors:
        # Simulate realistic factor returns
        returns = np.random.normal(0.004, 0.045, len(dates))
        factor_returns[f'{factor}_Long-Short'] = returns

    return factor_returns


def run_factor_model_analysis(
    factor_returns: pd.DataFrame,
    config: dict,
    skip: bool = False
) -> pd.DataFrame:
    """Run CAPM, FF3, FF5 regressions."""

    checkpoint_file = os.path.join(
        config['paths']['checkpoints'], 'factor_model_results.csv'
    )

    if skip or os.path.exists(checkpoint_file):
        logging.info("Loading factor model results from checkpoint...")
        return load_checkpoint('factor_model_results', config['paths']['checkpoints'])

    logging.info("Running factor model regressions...")

    try:
        from analysis.factor_models import load_fama_french_3factor, load_fama_french_5factor

        ff3_factors = load_fama_french_3factor()
        ff5_factors = load_fama_french_5factor()

        results = test_all_models(factor_returns, ff3_factors, ff5_factors)

    except Exception as e:
        logging.warning(f"Could not download Fama-French data: {e}")
        logging.warning("Generating placeholder factor model results")

        # Placeholder results
        results = pd.DataFrame()

    # Save checkpoint
    if not results.empty:
        save_checkpoint(results, 'factor_model_results', config['paths']['checkpoints'])

    return results


def run_sentiment_conditional_analysis(
    orthogonalized_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    config: dict
) -> dict:
    """Run sentiment-conditional returns analysis."""
    logging.info("Running sentiment-conditional analysis...")

    lags = config['regression']['newey_west_lags']['hml_test']
    portfolio_legs = config['sentiment_conditional'].get('portfolio_legs',
                                                          ['Long', 'Short', 'Long-Short'])

    results = sentiment_factor_analysis(
        sentiment_df=orthogonalized_df,
        residuals_df=residuals_df,
        portfolio_legs=portfolio_legs,
        lags=lags
    )

    # Save results
    for key, df in results.items():
        table = create_sentiment_conditional_table(df)
        output_path = os.path.join(
            config['paths']['tables'], f'sentiment_conditional_{key}.csv'
        )
        table.to_csv(output_path, index=False)

    logging.info(f"Saved {len(results)} sentiment-conditional result files")

    return results


def generate_figures(
    sentiment_df: pd.DataFrame,
    orthogonalized_df: pd.DataFrame,
    pca_results: dict,
    conditional_results: dict,
    config: dict
) -> None:
    """Generate all publication-ready figures."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    logging.info("Generating figures...")

    figures_path = config['paths']['figures']

    # 1. Correlation heatmap
    if not orthogonalized_df.empty:
        corr = orthogonalized_df.corr()
        fig = plot_correlation_heatmap(
            corr,
            title='Correlation Matrix: Orthogonalized Sentiment Indicators',
            save_path=os.path.join(figures_path, 'sentiment_correlation.png')
        )
        plt.close(fig)

    # 2. PCA loadings (first period available)
    for period_name, result in pca_results.items():
        if result is not None:
            period_safe = period_name.replace(' ', '_').lower()
            fig = plot_pca_loadings(
                result['loadings'],
                save_path=os.path.join(figures_path, f'pca_loadings_{period_safe}.png')
            )
            plt.close(fig)
            break  # Only plot first period for brevity

    # 3. HML t-statistics
    for key, df in list(conditional_results.items())[:3]:  # First 3 combinations
        fig = plot_hml_tstatistics(
            df,
            title=f'High-Minus-Low t-statistics: {key}',
            save_path=os.path.join(figures_path, f'hml_tstats_{key.replace("/", "_")}.png')
        )
        plt.close(fig)

    logging.info("Figures saved to results/figures/")


def main() -> None:
    """Main analysis pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        config['logging']['level'],
        config['logging']['file']
    )

    logging.info("=" * 60)
    logging.info("ASSET PRICING AND INVESTOR SENTIMENT ANALYSIS")
    logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)

    # Create output directories
    create_output_dirs(config)

    # ---------------------------------------------------------------
    # STEP 1: Load Sentiment Data
    # ---------------------------------------------------------------
    sentiment_df = load_sentiment_data(config)
    logging.info(f"Sentiment data loaded: {sentiment_df.shape}")

    # ---------------------------------------------------------------
    # STEP 2: Load Macro Data
    # ---------------------------------------------------------------
    macro_df = load_macro_data(config)
    logging.info(f"Macro data loaded: {macro_df.shape}")

    # ---------------------------------------------------------------
    # STEP 3: Orthogonalize Sentiment
    # ---------------------------------------------------------------
    skip_orth = args.skip_orthogonalization or args.load_checkpoints
    orthogonalized_df = run_orthogonalization(
        sentiment_df, macro_df, config, skip=skip_orth
    )
    logging.info(f"Orthogonalized sentiment: {orthogonalized_df.shape}")

    # ---------------------------------------------------------------
    # STEP 4: PCA Analysis
    # ---------------------------------------------------------------
    pca_results = run_pca_analysis(orthogonalized_df, config)

    # ---------------------------------------------------------------
    # STEP 5: Load Factor Data
    # ---------------------------------------------------------------
    factor_returns = load_factor_data(config)
    logging.info(f"Factor data loaded: {factor_returns.shape}")

    # ---------------------------------------------------------------
    # STEP 6: Factor Model Regressions
    # ---------------------------------------------------------------
    skip_models = args.skip_factor_models or args.load_checkpoints
    factor_model_results = run_factor_model_analysis(
        factor_returns, config, skip=skip_models
    )

    # ---------------------------------------------------------------
    # STEP 7: Sentiment-Conditional Analysis
    # ---------------------------------------------------------------
    # Use factor returns directly if no residuals available
    residuals_df = factor_returns

    conditional_results = run_sentiment_conditional_analysis(
        orthogonalized_df,
        residuals_df,
        config
    )

    # ---------------------------------------------------------------
    # STEP 8: Generate Figures
    # ---------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        generate_figures(
            sentiment_df,
            orthogonalized_df,
            pca_results,
            conditional_results,
            config
        )
    except Exception as e:
        logging.warning(f"Figure generation failed: {e}")

    logging.info("=" * 60)
    logging.info("ANALYSIS COMPLETE")
    logging.info(f"Results saved to: {config['paths']['results']}")
    logging.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
