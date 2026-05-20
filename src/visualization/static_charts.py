import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")


def _save(fig, output_dir: Path, stem: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_top10_by_citations(df: pd.DataFrame, output_dir: Path) -> None:
    top10 = (
        df.nlargest(10, "citation_count")[["title", "citation_count"]]
        .copy()
        .reset_index(drop=True)
    )
    top10["short_title"] = top10["title"].str[:55] + "…"

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top10["short_title"], top10["citation_count"], color=sns.color_palette("muted", 10))
    ax.set_xlabel("Citation Count")
    ax.set_title("Top 10 Papers by Citation Count")
    ax.invert_yaxis()

    for bar, val in zip(bars, top10["citation_count"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9)

    fig.tight_layout()
    _save(fig, output_dir, "top10_by_citations")


def plot_papers_per_year(df: pd.DataFrame, output_dir: Path) -> None:
    yearly = (
        df.groupby("published_year")
        .agg(paper_count=("paper_id", "count"), avg_relevance=("relevance_score", "mean"))
        .reset_index()
        .sort_values("published_year")
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(yearly["published_year"], yearly["paper_count"], color="#4C72B0", alpha=0.7, label="Paper Count")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax.twinx()
    ax2.plot(yearly["published_year"], yearly["avg_relevance"], color="#DD8452",
             marker="o", linewidth=2, label="Avg Relevance Score")
    ax2.set_ylabel("Average Relevance Score", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.set_title("Papers Published per Year with Average Relevance Score")

    fig.tight_layout()
    _save(fig, output_dir, "papers_per_year")


def plot_citation_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["citation_count"].dropna(), bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_xlabel("Citation Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Citation Counts")
    fig.tight_layout()
    _save(fig, output_dir, "citation_distribution")


def plot_citations_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    top_cats = df["primary_category"].value_counts().nlargest(8).index
    filtered = df[df["primary_category"].isin(top_cats)].copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.boxplot(
        data=filtered,
        x="primary_category",
        y="citation_count",
        hue="primary_category",
        legend=False,
        palette="muted",
        ax=ax,
    )
    ax.set_xlabel("Primary Category")
    ax.set_ylabel("Citation Count")
    ax.set_title("Citation Count Distribution by Category")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    _save(fig, output_dir, "citations_by_category")


def plot_relevance_vs_citations(df: pd.DataFrame, output_dir: Path) -> None:
    sample = df[["relevance_score", "citation_count", "primary_category"]].dropna()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=sample,
        x="relevance_score",
        y="citation_count",
        hue="primary_category",
        alpha=0.7,
        ax=ax,
        palette="tab10",
    )
    ax.set_xlabel("Relevance Score")
    ax.set_ylabel("Citation Count")
    ax.set_title("Relevance Score vs Citation Count")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "relevance_vs_citations")


def plot_category_paper_count(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["primary_category"].value_counts().reset_index()
    counts.columns = ["category", "count"]

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("muted", len(counts))
    ax.bar(counts["category"], counts["count"], color=palette)
    ax.set_xlabel("Primary Category")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Paper Count per Primary Category")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    _save(fig, output_dir, "category_paper_count")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout()
    _save(fig, output_dir, "correlation_heatmap")


def plot_dashboard_subplots(df: pd.DataFrame, output_dir: Path) -> None:
    top10 = df.nlargest(10, "citation_count")[["title", "citation_count"]].copy()
    top10["short_title"] = top10["title"].str[:40] + "…"

    yearly = (
        df.groupby("published_year")
        .agg(paper_count=("paper_id", "count"), avg_relevance=("relevance_score", "mean"))
        .reset_index()
        .sort_values("published_year")
    )

    counts = df["primary_category"].value_counts().reset_index()
    counts.columns = ["category", "count"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("ArXiv AI Detection Papers – Analytics Dashboard", fontsize=16, fontweight="bold")

    # Panel [0,0]: top 10 horizontal bar
    axes[0, 0].barh(top10["short_title"], top10["citation_count"], color="#4C72B0")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Top 10 by Citations")
    axes[0, 0].tick_params(axis="y", labelsize=7)

    # Panel [0,1]: papers per year
    axes[0, 1].bar(yearly["published_year"], yearly["paper_count"], color="#4C72B0", alpha=0.7)
    ax_twin = axes[0, 1].twinx()
    ax_twin.plot(yearly["published_year"], yearly["avg_relevance"], color="#DD8452", marker="o")
    axes[0, 1].set_title("Papers/Year & Avg Relevance")

    # Panel [1,0]: citation distribution
    axes[1, 0].hist(df["citation_count"].dropna(), bins=25, color="#55A868")
    axes[1, 0].set_title("Citation Distribution")
    axes[1, 0].set_xlabel("Citations")

    # Panel [1,1]: category counts
    axes[1, 1].bar(counts["category"], counts["count"], color=sns.color_palette("muted", len(counts)))
    axes[1, 1].set_title("Papers per Category")
    axes[1, 1].tick_params(axis="x", rotation=35, labelsize=7)

    fig.tight_layout()
    _save(fig, output_dir, "dashboard_subplots")
