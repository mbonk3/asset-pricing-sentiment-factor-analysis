"""
Time Series Visualization Module

This module provides functions for creating publication-quality plots
for sentiment analysis and asset pricing results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple


def plot_sentiment_indicator(
    series: pd.Series,
    title: str,
    ylabel: str = 'Value',
    color: str = '#1f77b4',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot a single sentiment indicator over time.

    Args:
        series: Time series data
        title: Plot title
        ylabel: Y-axis label
        color: Line color
        save_path: Path to save figure (None = don't save)
        figsize: Figure size (width, height) in inches

    Returns:
        Matplotlib Figure object

    Examples:
        >>> dates = pd.date_range('2000-01', periods=24, freq='M')
        >>> series = pd.Series(np.random.normal(0, 1, 24), index=dates)
        >>> fig = plot_sentiment_indicator(series, 'Test Sentiment')
        >>> isinstance(fig, plt.Figure)
        True
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(series.index, series.values, color=color, linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.fill_between(series.index, series.values, 0,
                    where=(series.values > 0), alpha=0.3, color='green', label='Positive')
    ax.fill_between(series.index, series.values, 0,
                    where=(series.values < 0), alpha=0.3, color='red', label='Negative')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(5))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_orthogonalized_comparison(
    original: pd.Series,
    orthogonalized: pd.Series,
    title: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot original vs orthogonalized sentiment indicator side-by-side.

    Args:
        original: Original sentiment series
        orthogonalized: Orthogonalized sentiment series
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Examples:
        >>> dates = pd.date_range('2000-01', periods=24, freq='M')
        >>> orig = pd.Series(np.random.normal(0, 1, 24), index=dates)
        >>> orth = pd.Series(np.random.normal(0, 0.8, 24), index=dates)
        >>> fig = plot_orthogonalized_comparison(orig, orth, 'Test')
        >>> isinstance(fig, plt.Figure)
        True
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Align data
    combined = pd.DataFrame({'Original': original, 'Orthogonalized': orthogonalized})

    for idx, (col, ax) in enumerate(zip(['Original', 'Orthogonalized'], axes)):
        series = combined[col].dropna()
        color = '#1f77b4' if col == 'Original' else '#ff7f0e'

        ax.plot(series.index, series.values, color=color, linewidth=1.2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values > 0), alpha=0.2, color='green')
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values < 0), alpha=0.2, color='red')

        ax.set_title(f'{col}', fontsize=12)
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True
) -> plt.Figure:
    """
    Plot a correlation matrix as a heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        annot: Whether to annotate cells with values

    Returns:
        Matplotlib Figure object

    Examples:
        >>> df = pd.DataFrame({'a': range(100), 'b': range(50, 150), 'c': range(100, 0, -1)})
        >>> corr = df.corr()
        >>> fig = plot_correlation_heatmap(corr, 'Test Correlation')
        >>> isinstance(fig, plt.Figure)
        True
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        ax=ax,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        annot=annot,
        fmt='.2f',
        annot_kws={'size': 9},
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_pca_loadings(
    loadings_df: pd.DataFrame,
    n_components: int = 3,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot PCA loadings as bar chart.

    Args:
        loadings_df: DataFrame with loadings (rows = indicators, columns = components)
        n_components: Number of components to plot
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Examples:
        >>> indicators = ['BW', 'VIX', 'ICS', 'AAII']
        >>> loadings = pd.DataFrame(
        ...     np.random.uniform(-1, 1, (4, 3)),
        ...     index=indicators,
        ...     columns=['PC1', 'PC2', 'PC3']
        ... )
        >>> fig = plot_pca_loadings(loadings)
        >>> isinstance(fig, plt.Figure)
        True
    """
    components = [f'PC{i+1}' for i in range(min(n_components, loadings_df.shape[1]))]
    components = [c for c in components if c in loadings_df.columns]

    fig, axes = plt.subplots(1, len(components), figsize=figsize)

    if len(components) == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (comp, ax) in enumerate(zip(components, axes)):
        loadings = loadings_df[comp].sort_values()
        bar_colors = ['#d62728' if v < 0 else '#2ca02c' for v in loadings.values]

        ax.barh(loadings.index, loadings.values, color=bar_colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.8)
        ax.set_title(comp, fontsize=12, fontweight='bold')
        ax.set_xlabel('Loading')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (label, value) in enumerate(zip(loadings.index, loadings.values)):
            ax.text(value + 0.01 * np.sign(value), i, f'{value:.2f}',
                    va='center', ha='left' if value > 0 else 'right', fontsize=8)

    fig.suptitle('PCA Loadings by Component', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_explained_variance(
    variance_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot explained variance scree plot.

    Args:
        variance_df: Output from calculate_explained_variance
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Examples:
        >>> import pandas as pd
        >>> var_df = pd.DataFrame({
        ...     'Component': ['PC1', 'PC2', 'PC3'],
        ...     'Explained_Variance_Ratio': [0.4, 0.25, 0.15],
        ...     'Cumulative_Variance': [0.4, 0.65, 0.80]
        ... })
        >>> fig = plot_explained_variance(var_df)
        >>> isinstance(fig, plt.Figure)
        True
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Individual variance
    ax1.bar(variance_df['Component'], variance_df['Explained_Variance_Ratio'] * 100,
            color='#1f77b4', alpha=0.7, label='Individual')
    ax1.set_ylabel('Explained Variance (%)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Cumulative variance
    ax2.plot(variance_df['Component'], variance_df['Cumulative_Variance'] * 100,
             color='#d62728', marker='o', linewidth=2, label='Cumulative')
    ax2.set_ylabel('Cumulative Explained Variance (%)', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.axhline(y=80, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    ax1.set_xlabel('Principal Component')
    ax1.set_title('Explained Variance - Scree Plot', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3, axis='y')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_hml_tstatistics(
    hml_df: pd.DataFrame,
    title: str = 'High-Minus-Low t-statistics',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot High-Minus-Low t-statistics as horizontal bar chart.

    Args:
        hml_df: DataFrame with Factor and HML_tstat columns
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Examples:
        >>> hml = pd.DataFrame({
        ...     'Factor': ['momentum', 'value', 'size'],
        ...     'HML_tstat': [2.5, -1.2, 0.8]
        ... })
        >>> fig = plot_hml_tstatistics(hml)
        >>> isinstance(fig, plt.Figure)
        True
    """
    # Sort by t-statistic
    hml_sorted = hml_df.sort_values('HML_tstat', ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#2ca02c' if t > 1.96 else '#d62728' if t < -1.96 else '#7f7f7f'
              for t in hml_sorted['HML_tstat']]

    bars = ax.barh(hml_sorted['Factor'], hml_sorted['HML_tstat'],
                   color=colors, alpha=0.8)

    # Significance lines
    ax.axvline(x=1.96, color='black', linestyle='--', linewidth=1,
               label='5% significance', alpha=0.8)
    ax.axvline(x=-1.96, color='black', linestyle='--', linewidth=1, alpha=0.8)
    ax.axvline(x=2.576, color='black', linestyle=':', linewidth=0.8,
               label='1% significance', alpha=0.7)
    ax.axvline(x=-2.576, color='black', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.set_xlabel('t-statistic (Newey-West, 3 lags)', fontsize=11)
    ax.set_ylabel('Factor')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='x')

    # Add significance count
    n_sig = (abs(hml_sorted['HML_tstat']) > 1.96).sum()
    ax.text(0.02, 0.98, f'Significant: {n_sig}/{len(hml_sorted)}',
            transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_alpha_survival(
    survival_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot alpha survival across factor models (CAPM, FF3, FF5).

    Args:
        survival_df: Output from calculate_alpha_survival
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Matplotlib Figure object

    Examples:
        >>> survival = pd.DataFrame({
        ...     'factor': ['mom', 'value', 'size'],
        ...     'CAPM_significant': [True, True, False],
        ...     'FF3_significant': [True, False, False],
        ...     'FF5_significant': [True, False, False]
        ... })
        >>> fig = plot_alpha_survival(survival)
        >>> isinstance(fig, plt.Figure)
        True
    """
    models = ['CAPM', 'FF3', 'FF5']
    sig_cols = [f'{m}_significant' for m in models]
    available = [c for c in sig_cols if c in survival_df.columns]

    counts = [survival_df[col].sum() for col in available]
    total = len(survival_df)

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        [m for m, c in zip(models, sig_cols) if c in survival_df.columns],
        counts,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        alpha=0.8,
        edgecolor='white'
    )

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{count}/{total}\n({count/total:.0%})',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel('Factor Model', fontsize=12)
    ax.set_ylabel('Number of Significant Anomalies', fontsize=12)
    ax.set_title('Alpha Survival Across Factor Models (p < 5%)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, total * 1.15)
    ax.axhline(y=total * 0.05, color='gray', linestyle='--', linewidth=0.8,
               alpha=0.7, label='5% of total (chance)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
