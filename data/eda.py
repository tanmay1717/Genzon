"""
Genzon — EDA Helper Functions
Reusable plotting and analysis utilities for notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# ─── Style defaults ─────────────────────────────────────────

COLORS = {"genuine": "#2ecc71", "fake": "#e74c3c"}
LABEL_MAP = {0: "Genuine", 1: "Fake"}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["figure.dpi"] = 100


# ─── Distribution plots ─────────────────────────────────────


def plot_class_distribution(df: pd.DataFrame, label_col: str = "label_encoded"):
    """Bar chart of class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    counts = df[label_col].value_counts()
    labels = [LABEL_MAP.get(i, str(i)) for i in counts.index]
    colors = [COLORS.get(l.lower(), "#95a5a6") for l in labels]

    # Count plot
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Class distribution (count)")
    axes[0].set_ylabel("Number of reviews")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

    # Percentage plot
    pcts = counts.values / counts.values.sum() * 100
    axes[1].bar(labels, pcts, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Class distribution (%)")
    axes[1].set_ylabel("Percentage")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(pcts):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    label_col: str = "label_encoded",
    bins: int = 50,
):
    """Overlapping histograms for a feature split by label."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for label_val, label_name in LABEL_MAP.items():
        subset = df[df[label_col] == label_val][feature].dropna()
        ax.hist(
            subset,
            bins=bins,
            alpha=0.6,
            label=label_name,
            color=COLORS[label_name.lower()],
            density=True,
        )

    ax.set_title(f"Distribution of '{feature}' by class")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_boxplots(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "label_encoded",
):
    """Side-by-side boxplots for multiple features."""
    n = len(features)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(features):
        df_plot = df[[feat, label_col]].copy()
        df_plot["label_name"] = df_plot[label_col].map(LABEL_MAP)
        sns.boxplot(
            data=df_plot,
            x="label_name",
            y=feat,
            palette=COLORS,
            ax=axes[i],
        )
        axes[i].set_title(feat)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ─── Text analysis ──────────────────────────────────────────


def plot_word_clouds(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label_encoded",
):
    """Word clouds for fake vs genuine reviews."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (label_val, label_name) in zip(axes, LABEL_MAP.items()):
        text = " ".join(df[df[label_col] == label_val][text_col].dropna().astype(str))
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="Greens" if label_name == "Genuine" else "Reds",
            max_words=150,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{label_name} reviews", fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_review_length_dist(
    df: pd.DataFrame,
    text_col: str = "text_clean",
    label_col: str = "label_encoded",
):
    """Review length distribution (word count)."""
    df_temp = df.copy()
    df_temp["_wc"] = df_temp[text_col].str.split().str.len()

    fig, ax = plt.subplots(figsize=(10, 4))
    for label_val, label_name in LABEL_MAP.items():
        subset = df_temp[df_temp[label_col] == label_val]["_wc"]
        ax.hist(
            subset,
            bins=80,
            alpha=0.6,
            label=f"{label_name} (median={subset.median():.0f})",
            color=COLORS[label_name.lower()],
            density=True,
        )
    ax.set_title("Review length distribution (word count)")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_xlim(0, df_temp["_wc"].quantile(0.99))
    plt.tight_layout()
    plt.show()


# ─── Correlation ────────────────────────────────────────────


def plot_feature_correlations(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "label_encoded",
):
    """Correlation heatmap of numeric features with the label."""
    cols = features + [label_col]
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    plt.show()


# ─── Summary statistics ─────────────────────────────────────


def print_summary_stats(
    df: pd.DataFrame,
    features: list[str],
    label_col: str = "label_encoded",
):
    """Print mean ± std for each feature grouped by class."""
    print(f"{'Feature':<25} {'Genuine (mean±std)':<25} {'Fake (mean±std)':<25} {'Diff':<10}")
    print("-" * 85)

    for feat in features:
        if feat not in df.columns:
            continue

        genuine = df[df[label_col] == 0][feat]
        fake = df[df[label_col] == 1][feat]

        g_str = f"{genuine.mean():.3f} ± {genuine.std():.3f}"
        f_str = f"{fake.mean():.3f} ± {fake.std():.3f}"
        diff = fake.mean() - genuine.mean()
        sign = "+" if diff > 0 else ""

        print(f"{feat:<25} {g_str:<25} {f_str:<25} {sign}{diff:.3f}")