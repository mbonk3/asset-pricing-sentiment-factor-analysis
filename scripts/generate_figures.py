"""Generate all figures for the GitHub Pages site."""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"C:\Users\mbonk002\OneDrive - PwC\09_Master\Thesis\Auswertungen"
OUT  = os.path.join(REPO, "results", "figures")
os.makedirs(OUT, exist_ok=True)

# ── design tokens ──────────────────────────────────────────────────────────────
NAVY    = "#1A3A5C"
BLUE    = "#0071E3"
SILVER  = "#F5F5F7"
DARK    = "#1D1D1F"
MID     = "#6E6E73"
WHITE   = "#FFFFFF"
RED     = "#C0392B"
GREEN   = "#27AE60"
AMBER   = "#E67E22"

PALETTE = [BLUE, NAVY, GREEN, AMBER, RED, "#8E44AD", "#16A085"]
SENT_COLORS = {
    "BW": BLUE, "Investor": NAVY, "ICS": GREEN,
    "AAII": AMBER, "VIX": RED, "Manager": "#8E44AD",
    "Employee": "#16A085", "BehavioralSentimentIndex": DARK,
}

def setup():
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.linewidth":   0.8,
        "axes.labelcolor":  DARK,
        "axes.titlecolor":  DARK,
        "xtick.color":      MID,
        "ytick.color":      MID,
        "text.color":       DARK,
        "figure.facecolor": WHITE,
        "axes.facecolor":   WHITE,
        "grid.color":       "#E8E8ED",
        "grid.linewidth":   0.6,
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
    })

setup()

# ── 1. PCA Scree Plot ──────────────────────────────────────────────────────────
def fig_pca_scree():
    df = pd.read_excel(os.path.join(DATA, "Sentiment Analysis", "explained_variance_all.xlsx"))
    samples = {
        "Long Sample\n(BW, Investor, ICS · 1978–2023)": "Long Sample (BW, Investor, ICS)",
        "Modern Sample\n(+VIX & AAII · 1990–2023)": "Modern Sample (adds VIX & AAII)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(WHITE)

    for ax, (title, key) in zip(axes, samples.items()):
        sub = df[df["Sample"].str.startswith(key.split("(")[0].strip())].copy()
        if sub.empty:
            sub = df[df["Sample"] == key]
        bars = ax.bar(sub["PC"].astype(str), sub["Explained Variance Ratio"] * 100,
                      color=BLUE, width=0.55, zorder=3)
        ax.plot(sub["PC"].astype(str), sub["Cumulative Variance"] * 100,
                color=NAVY, marker="o", linewidth=2, markersize=6, zorder=4, label="Cumulative")
        for bar, val in zip(bars, sub["Explained Variance Ratio"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, color=DARK)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Principal Component", fontsize=10)
        ax.set_ylabel("Explained Variance (%)", fontsize=10)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=9, frameon=False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("PCA of Orthogonalized Sentiment Indicators", fontsize=14,
                 fontweight="bold", y=1.02, color=DARK)
    fig.tight_layout()
    path = os.path.join(OUT, "pca_scree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {path}")

# ── 2. Correlation Heatmap ─────────────────────────────────────────────────────
def fig_correlation_heatmap():
    df = pd.read_excel(os.path.join(DATA, "Sentiment Analysis", "PairwiseCorrelations.xlsx"),
                       index_col=0)
    labels = list(df.columns)

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(df.values, dtype=bool), k=1)

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(df.values, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                center=0, vmin=-1, vmax=1,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="#E8E8ED",
                annot_kws={"size": 10}, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})
    ax.set_title("Pairwise Correlations of Orthogonalized Sentiment Indicators",
                 fontsize=12, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    fig.tight_layout()
    path = os.path.join(OUT, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {path}")

# ── 3. BSI Time Series ─────────────────────────────────────────────────────────
def fig_bsi_timeseries():
    df = pd.read_excel(os.path.join(DATA, "Sentiment Analysis", "BSI Comparison.xlsx"),
                       sheet_name="StandardizedSentiments", index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)

    fig, ax = plt.subplots(figsize=(12, 5))
    cols_background = [c for c in df.columns if c != "BehavioralSentimentIndex"]
    for col in cols_background:
        color = SENT_COLORS.get(col, MID)
        ax.plot(df.index, df[col], color=color, linewidth=0.9, alpha=0.45, label=col)

    ax.plot(df.index, df["BehavioralSentimentIndex"], color=DARK,
            linewidth=2.2, label="BSI (Composite)", zorder=5)
    ax.axhline(0, color=MID, linewidth=0.8, linestyle="--", alpha=0.7)

    # shade recessions roughly (NBER: 2001, 2008-09, 2020)
    for start, end in [("2001-03", "2001-11"), ("2007-12", "2009-06"), ("2020-02", "2020-04")]:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=RED, alpha=0.08)

    ax.set_title("Behavioral Sentiment Index vs. Individual Sentiment Indicators (Standardized)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Standardized Score (z-score)", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    note = "Shaded areas: NBER recession periods"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=MID, style="italic")

    fig.tight_layout()
    path = os.path.join(OUT, "bsi_timeseries.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {path}")

# ── 4. Alpha Survival Bar Chart ────────────────────────────────────────────────
def fig_alpha_survival():
    path_file = os.path.join(DATA, "Anomaly Returns", "df_alpha_survival.xlsx")
    rows = []
    for model in ["CAPM", "FF3", "FF5"]:
        df = pd.read_excel(path_file, sheet_name=model)
        total = len(df)
        sig   = int(df["Significant"].sum())
        rob   = int(df["Robust"].sum())
        rows.append({"Model": model, "Total": total,
                     "Significant": sig, "Robust": rob,
                     "Pct_Sig": sig / total * 100, "Pct_Rob": rob / total * 100})
    res = pd.DataFrame(rows)

    x = np.arange(len(res))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x - w/2, res["Pct_Sig"], w, label="Significant (p<0.05)", color=BLUE, zorder=3)
    b2 = ax.bar(x + w/2, res["Pct_Rob"], w, label="Robust (sig. across all models)", color=NAVY, zorder=3)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, color=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(["CAPM", "Fama-French 3-Factor", "Fama-French 5-Factor"], fontsize=11)
    ax.set_ylabel("Share of Anomaly Portfolios (%)", fontsize=10)
    ax.set_ylim(0, 90)
    ax.set_title("Anomaly Alpha Survival Across Asset Pricing Models\n"
                 f"(n = {res['Total'].iloc[0]:,} portfolios across OAP & JKP)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(OUT, "alpha_survival.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {out}")

# ── 5. Sentiment-Conditional Returns (OAP Robust) ─────────────────────────────
def fig_sentiment_conditional():
    all_files = glob.glob(os.path.join(DATA, "General Anomalies X Sentiment", "*.xlsx"))
    dfs = [pd.read_excel(f) for f in all_files]
    combined = pd.concat(dfs, ignore_index=True)

    # Focus on OAP Robust, CAPM model (clearest signal)
    sub = combined[
        (combined["Residual_Group"] == "Robust") &
        (combined["Country"] == "OAP") &
        (combined["Model"] == "CAPM")
    ].drop_duplicates(subset=["Sentiment_Indicator"]).copy()

    sub = sub.sort_values("Mean_Residual_HighMinusLow", ascending=True)
    indicators = sub["Sentiment_Indicator"].tolist()
    hml  = sub["Mean_Residual_HighMinusLow"].values
    high = sub["Mean_Residual_High"].values
    low  = sub["Mean_Residual_Low"].values

    colors_bar = [GREEN if v > 0 else RED for v in hml]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: H-M-L
    ax = axes[0]
    bars = ax.barh(indicators, hml, color=colors_bar, height=0.55, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.9, linestyle="-")
    for bar, val in zip(bars, hml):
        pad = 0.01 if val >= 0 else -0.01
        ha  = "left" if val >= 0 else "right"
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}%", va="center", ha=ha, fontsize=8.5, color=DARK)
    ax.set_xlabel("Monthly Return Difference (High − Low Sentiment, %)", fontsize=10)
    ax.set_title("High-Minus-Low Sentiment Effect\n(OAP Robust Anomalies, CAPM Alpha)",
                 fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)

    # right: High vs Low side-by-side
    ax2 = axes[1]
    y  = np.arange(len(indicators))
    w  = 0.35
    ax2.barh(y - w/2, high, w, color=BLUE,   label="High Sentiment", zorder=3)
    ax2.barh(y + w/2, low,  w, color=RED,    label="Low Sentiment",  zorder=3)
    ax2.axvline(0, color=DARK, linewidth=0.9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(indicators, fontsize=10)
    ax2.set_xlabel("Mean Monthly Alpha (%)", fontsize=10)
    ax2.set_title("Mean Alpha in High vs. Low Sentiment Regimes\n(OAP Robust Anomalies, CAPM Alpha)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10, frameon=False)
    ax2.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.yaxis.grid(False)

    fig.tight_layout(pad=2)
    out = os.path.join(OUT, "sentiment_conditional.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {out}")

# ── 6. FDR Significant Correlations Network / Summary ─────────────────────────
def fig_fdr_summary():
    df = pd.read_excel(os.path.join(DATA, "Sentiment Analysis", "fdr_results_all.xlsx"))
    # Use modern sample (most complete)
    modern = df[df["Sample"] == "Modern Sample (adds VIX & AAII)"].copy()
    modern["Significant"] = modern["Significant (5%)"]
    modern["Label"] = modern["Variable 1"] + " – " + modern["Variable 2"]
    modern = modern.sort_values("Correlation")

    colors = [GREEN if (row.Significant and row.Correlation > 0)
              else RED if (row.Significant and row.Correlation < 0)
              else "#CCCCCC"
              for _, row in modern.iterrows()]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(modern["Label"], modern["Correlation"], color=colors, height=0.6, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.9)
    for bar, val in zip(bars, modern["Correlation"]):
        pad = 0.005 if val >= 0 else -0.005
        ha  = "left" if val >= 0 else "right"
        ax.text(val + pad, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha=ha, fontsize=8.5, color=DARK)
    ax.set_xlabel("Pearson Correlation (Orthogonalized Sentiment)", fontsize=10)
    ax.set_title("Pairwise Correlations with FDR Correction\n"
                 "Modern Sample (1990–2023, 5 indicators)", fontsize=11, fontweight="bold")
    patches = [
        mpatches.Patch(color=GREEN, label="Significant positive"),
        mpatches.Patch(color=RED,   label="Significant negative"),
        mpatches.Patch(color="#CCCCCC", label="Not significant"),
    ]
    ax.legend(handles=patches, fontsize=9, frameon=False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(OUT, "fdr_correlations.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {out}")

# ── 7. PCA Loadings Heatmap (Long Sample) ─────────────────────────────────────
def fig_pca_loadings():
    raw = pd.read_excel(os.path.join(DATA, "Sentiment Analysis", "loadings_all.xlsx"))
    # Long sample is rows 0-2
    long = raw.iloc[0:3][["Sentiment", "PC1", "PC2", "PC3"]].copy()
    long = long.set_index("Sentiment")

    ev_long = [0.562374, 0.332029, 0.105596]
    col_labels = [f"PC1\n({ev_long[0]*100:.1f}%)",
                  f"PC2\n({ev_long[1]*100:.1f}%)",
                  f"PC3\n({ev_long[2]*100:.1f}%)"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(long.values.astype(float), annot=True, fmt=".2f", cmap=cmap,
                center=0, vmin=-1, vmax=1,
                xticklabels=col_labels,
                yticklabels=long.index.tolist(),
                linewidths=0.5, linecolor="#E8E8ED",
                annot_kws={"size": 11}, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Loading"})
    ax.set_title("PCA Factor Loadings – Long Sample (1978–2023)\nBW, Investor, ICS",
                 fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "pca_loadings.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"Saved {out}")

# ── run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig_pca_scree()
    fig_pca_loadings()
    fig_correlation_heatmap()
    fig_bsi_timeseries()
    fig_alpha_survival()
    fig_sentiment_conditional()
    fig_fdr_summary()
    print("Done. All figures saved to:", OUT)
