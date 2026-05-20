"""
CLI entry-point for chart generation.
Usage:
    python scripts/generate_visualizations.py
    python scripts/generate_visualizations.py --data path/to/cleaned.csv
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.visualization.chart_generator import generate_all_charts


def main():
    parser = argparse.ArgumentParser(description="Generate all Lab 12 visualizations.")
    parser.add_argument(
        "--data",
        default="data/processed/cleaned/cleaned_data.csv",
        help="Path to the cleaned CSV dataset.",
    )
    parser.add_argument(
        "--static-out",
        default="outputs/visualizations/static",
        help="Output directory for static PNG/PDF charts.",
    )
    parser.add_argument(
        "--interactive-out",
        default="outputs/visualizations/interactive",
        help="Output directory for interactive HTML charts.",
    )
    args = parser.parse_args()

    generate_all_charts(
        data_path=args.data,
        static_out=args.static_out,
        interactive_out=args.interactive_out,
    )


if __name__ == "__main__":
    main()
