"""
Step 3: Exploratory Data Analysis.

Reads the full engineered feature table (data/processed/features_full.csv) and
produces summary figures into figures/. Uses the full table (more columns, richer
story) rather than the 25-feature modeling subset, per the assignment's "Data
Understanding" section.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IN_PATH = Path("data/processed/features_full.csv")
FIG_DIR = Path("figures")
TARGET = "ai_f2a2"

sns.set_theme(style="whitegrid")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)
    modeling = df[df[TARGET].notna()].copy()

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(modeling[TARGET], bins=30, kde=True, ax=ax, color="#33C7B0")
    ax.set_title("Distribution of Expected Productivity Increase (%)")
    ax.set_xlabel("ai_f2a2 -- expected productivity increase (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Missingness overview (full table)
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(25)
    fig, ax = plt.subplots(figsize=(8, 8))
    miss.iloc[::-1].plot(kind="barh", ax=ax, color="#F0A83A")
    ax.set_xlabel("Fraction missing")
    ax.set_title("Top 25 Columns by Missingness (full engineered table)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_missingness.png", dpi=150)
    plt.close(fig)

    # 3. Correlation heatmap: numeric features vs target
    numeric = modeling.select_dtypes(include=[np.number])
    corr_with_target = numeric.corr()[TARGET].drop(TARGET).sort_values()
    top_corr = pd.concat([corr_with_target.head(10), corr_with_target.tail(10)])
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#33C7B0" if v > 0 else "#F0A83A" for v in top_corr.values]
    top_corr.plot(kind="barh", ax=ax, color=colors)
    ax.set_xlabel("Correlation with target")
    ax.set_title("Top +/- 10 Correlations with Expected Productivity Increase")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_target_correlations.png", dpi=150)
    plt.close(fig)

    # 4. AI adoption vs target
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(
        data=modeling, x="ai_adopter", y=TARGET, ax=ax,
        palette=["#7B82AC", "#33C7B0"],
    )
    ax.set_xticklabels(["Non-adopter", "AI adopter"])
    ax.set_title("Expected Productivity Increase by AI Adoption Status")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_target_by_adoption.png", dpi=150)
    plt.close(fig)

    # 5. Firm size vs target
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=modeling, x="log_workers", y=TARGET, hue="ai_adopter",
        palette={0: "#7B82AC", 1: "#33C7B0"}, ax=ax, alpha=0.6,
    )
    ax.set_title("Firm Size vs Expected Productivity Increase")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_size_vs_target.png", dpi=150)
    plt.close(fig)

    # Summary stats table
    summary = modeling.describe().T
    summary.to_csv(Path("reports") / "eda_summary_stats.csv")

    print("EDA figures written to figures/. Summary stats -> reports/eda_summary_stats.csv")


if __name__ == "__main__":
    main()
