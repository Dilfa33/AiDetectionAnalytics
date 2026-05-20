from pathlib import Path
import pandas as pd

from .static_charts import (
    plot_top10_by_citations,
    plot_papers_per_year,
    plot_citation_distribution,
    plot_citations_by_category,
    plot_relevance_vs_citations,
    plot_category_paper_count,
    plot_correlation_heatmap,
    plot_dashboard_subplots,
)
from .interactive_charts import (
    interactive_citations_scatter,
    interactive_category_bar,
    interactive_yearly_trend,
    interactive_citation_box,
    interactive_multi_layout,
)

DEFAULT_DATA = Path("data/processed/cleaned/cleaned_data.csv")
DEFAULT_STATIC = Path("outputs/visualizations/static")
DEFAULT_INTERACTIVE = Path("outputs/visualizations/interactive")

STATIC_CHARTS = [
    ("plot_top10_by_citations",    plot_top10_by_citations),
    ("plot_papers_per_year",       plot_papers_per_year),
    ("plot_citation_distribution", plot_citation_distribution),
    ("plot_citations_by_category", plot_citations_by_category),
    ("plot_relevance_vs_citations",plot_relevance_vs_citations),
    ("plot_category_paper_count",  plot_category_paper_count),
    ("plot_correlation_heatmap",   plot_correlation_heatmap),
    ("plot_dashboard_subplots",    plot_dashboard_subplots),
]

INTERACTIVE_CHARTS = [
    ("interactive_citations_scatter", interactive_citations_scatter),
    ("interactive_category_bar",      interactive_category_bar),
    ("interactive_yearly_trend",      interactive_yearly_trend),
    ("interactive_citation_box",      interactive_citation_box),
    ("interactive_multi_layout",      interactive_multi_layout),
]


def generate_all_charts(
    data_path=None,
    static_out=None,
    interactive_out=None,
    df: pd.DataFrame = None,
):
    static_out = Path(static_out or DEFAULT_STATIC)
    interactive_out = Path(interactive_out or DEFAULT_INTERACTIVE)
    static_out.mkdir(parents=True, exist_ok=True)
    interactive_out.mkdir(parents=True, exist_ok=True)

    if df is None:
        data_path = Path(data_path or DEFAULT_DATA)
        print(f"[chart_generator] Loading data from {data_path}")
        df = pd.read_csv(data_path)

    print(f"[chart_generator] Dataset: {len(df)} rows, {len(df.columns)} columns")

    total = len(STATIC_CHARTS) + len(INTERACTIVE_CHARTS)
    done = 0

    print("\n── Static charts ──────────────────────────────────")
    for name, fn in STATIC_CHARTS:
        try:
            fn(df, static_out)
            done += 1
            print(f"  [{done}/{total}] ✓ {name}")
        except Exception as exc:
            print(f"  [{done}/{total}] ✗ {name}: {exc}")

    print("\n── Interactive charts ─────────────────────────────")
    for name, fn in INTERACTIVE_CHARTS:
        try:
            fn(df, interactive_out)
            done += 1
            print(f"  [{done}/{total}] ✓ {name}")
        except Exception as exc:
            print(f"  [{done}/{total}] ✗ {name}: {exc}")

    print(f"\n[chart_generator] Done. Static → {static_out}  Interactive → {interactive_out}")
