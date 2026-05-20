import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _save_html(fig, output_dir: Path, stem: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_dir / f"{stem}.html"))


def interactive_citations_scatter(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    data = df[["title", "relevance_score", "citation_count", "primary_category", "published_year"]].dropna()

    fig = px.scatter(
        data,
        x="relevance_score",
        y="citation_count",
        color="primary_category",
        hover_data={"title": True, "primary_category": True, "published_year": True,
                    "citation_count": True, "relevance_score": ":.3f"},
        title="Relevance Score vs Citation Count (Interactive)",
        labels={"relevance_score": "Relevance Score", "citation_count": "Citation Count",
                "primary_category": "Category"},
        opacity=0.75,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=8))
    _save_html(fig, output_dir, "citations_scatter")
    return fig


def interactive_category_bar(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    counts = (
        df.groupby("primary_category")
        .agg(paper_count=("paper_id", "count"),
             avg_citations=("citation_count", "mean"),
             avg_relevance=("relevance_score", "mean"))
        .reset_index()
        .sort_values("paper_count", ascending=False)
    )

    fig = px.bar(
        counts,
        x="primary_category",
        y="paper_count",
        color="primary_category",
        hover_data={"paper_count": True, "avg_citations": ":.1f", "avg_relevance": ":.3f"},
        title="Paper Count per Category (Interactive)",
        labels={"primary_category": "Category", "paper_count": "Number of Papers"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    _save_html(fig, output_dir, "category_bar")
    return fig


def interactive_yearly_trend(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    yearly = (
        df.groupby("published_year")
        .agg(paper_count=("paper_id", "count"),
             avg_citations=("citation_count", "mean"),
             avg_relevance=("relevance_score", "mean"))
        .reset_index()
        .sort_values("published_year")
    )

    fig = px.line(
        yearly,
        x="published_year",
        y="paper_count",
        markers=True,
        hover_data={"published_year": True, "paper_count": True,
                    "avg_citations": ":.1f", "avg_relevance": ":.3f"},
        title="Papers Published per Year (Interactive)",
        labels={"published_year": "Year", "paper_count": "Number of Papers"},
        template="plotly_white",
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=8))
    _save_html(fig, output_dir, "yearly_trend")
    return fig


def interactive_citation_box(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    top_cats = df["primary_category"].value_counts().nlargest(8).index
    filtered = df[df["primary_category"].isin(top_cats)].copy()

    fig = px.box(
        filtered,
        x="primary_category",
        y="citation_count",
        color="primary_category",
        hover_data={"title": True, "citation_count": True, "published_year": True},
        title="Citation Count by Category (Interactive)",
        labels={"primary_category": "Category", "citation_count": "Citation Count"},
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-25)
    _save_html(fig, output_dir, "citation_box")
    return fig


def interactive_multi_layout(df: pd.DataFrame, output_dir: Path) -> go.Figure:
    # Precompute aggregations
    yearly = (
        df.groupby("published_year")
        .agg(paper_count=("paper_id", "count"), avg_citations=("citation_count", "mean"))
        .reset_index()
        .sort_values("published_year")
    )
    cat_counts = df["primary_category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]

    top10 = df.nlargest(10, "citation_count")[["title", "citation_count"]].copy()
    top10["short_title"] = top10["title"].str[:45] + "…"

    top_cats = df["primary_category"].value_counts().nlargest(6).index
    box_data = df[df["primary_category"].isin(top_cats)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Relevance vs Citations",
            "Papers per Year",
            "Top 10 by Citations",
            "Citation Box by Category",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    # [1,1] Scatter
    fig.add_trace(
        go.Scatter(
            x=df["relevance_score"],
            y=df["citation_count"],
            mode="markers",
            marker=dict(size=6, opacity=0.6, color=df["citation_count"],
                        colorscale="Viridis", showscale=False),
            text=df["title"].str[:60],
            hovertemplate="<b>%{text}</b><br>Relevance: %{x:.3f}<br>Citations: %{y}<extra></extra>",
            name="Papers",
        ),
        row=1, col=1,
    )

    # [1,2] Line
    fig.add_trace(
        go.Scatter(
            x=yearly["published_year"],
            y=yearly["paper_count"],
            mode="lines+markers",
            line=dict(width=2),
            hovertemplate="Year: %{x}<br>Papers: %{y}<extra></extra>",
            name="Papers/Year",
        ),
        row=1, col=2,
    )

    # [2,1] Horizontal bar
    fig.add_trace(
        go.Bar(
            x=top10["citation_count"],
            y=top10["short_title"],
            orientation="h",
            hovertemplate="%{y}<br>Citations: %{x}<extra></extra>",
            name="Top 10",
            marker_color="#4C72B0",
        ),
        row=2, col=1,
    )

    # [2,2] Box per category
    for cat in top_cats:
        cat_df = box_data[box_data["primary_category"] == cat]
        fig.add_trace(
            go.Box(
                y=cat_df["citation_count"],
                name=cat,
                hovertemplate=f"Category: {cat}<br>Citations: %{{y}}<extra></extra>",
                showlegend=False,
            ),
            row=2, col=2,
        )

    fig.update_layout(
        title_text="ArXiv AI Detection Papers – Interactive Dashboard",
        height=900,
        template="plotly_white",
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)

    _save_html(fig, output_dir, "multi_layout_dashboard")
    return fig
