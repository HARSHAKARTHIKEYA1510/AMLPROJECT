"""
run_feature_pipeline.py - Orchestration script for M5 Feature Engineering Pipeline.

CSAI Student Project 5:
Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models.

Usage:
    python3 src/run_feature_pipeline.py [OPTIONS]

Example:
    python3 src/run_feature_pipeline.py --n_series 50 --selection_strategy diverse
    python3 src/run_feature_pipeline.py --n_series 100 --store_id CA_1 --val_days 28
"""

import sys
import argparse
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import M5FeatureEngineer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run M5 Feature Engineering & Chronological Train/Val Splitting Pipeline."
    )
    parser.add_argument(
        "--n_series",
        type=int,
        default=50,
        help="Number of series to subset for development (default: 50, use 0 for all 30,490 series).",
    )
    parser.add_argument(
        "--store_id",
        type=str,
        default=None,
        help="Optional store ID to filter by (e.g. CA_1, TX_2, WI_3).",
    )
    parser.add_argument(
        "--cat_id",
        type=str,
        default=None,
        help="Optional category ID to filter by (e.g. HOBBIES, HOUSEHOLD, FOODS).",
    )
    parser.add_argument(
        "--dept_id",
        type=str,
        default=None,
        help="Optional department ID to filter by (e.g. FOODS_1, HOBBIES_2).",
    )
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["diverse", "top_volume", "sample", "first"],
        default="diverse",
        help="Strategy to subset series when n_series < total pool (default: diverse).",
    )
    parser.add_argument(
        "--val_days",
        type=int,
        default=28,
        help="Chronological validation horizon in days (default: 28).",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 7, 28],
        help="Lag intervals to compute (default: 1 7 28).",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[7, 28],
        help="Rolling window sizes in days (default: 7 28).",
    )
    parser.add_argument(
        "--keep_warmup",
        action="store_true",
        help="If set, retains initial warmup rows with NaN features in training set (default: False).",
    )
    parser.add_argument(
        "--raw_sales",
        type=str,
        default="data/raw/sales_train_validation.csv",
        help="Path to raw sales CSV.",
    )
    parser.add_argument(
        "--calendar",
        type=str,
        default="data/raw/calendar.csv",
        help="Path to calendar CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Destination directory for processed files.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible series selection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    n_series = args.n_series if args.n_series > 0 else None
    store_ids = [args.store_id] if args.store_id else None
    cat_ids = [args.cat_id] if args.cat_id else None
    dept_ids = [args.dept_id] if args.dept_id else None

    engineer = M5FeatureEngineer(
        raw_sales_path=args.raw_sales,
        calendar_path=args.calendar,
        processed_dir=args.output_dir,
        lags=args.lags,
        rolling_windows=args.windows,
        validation_days=args.val_days,
        drop_warmup=(not args.keep_warmup),
    )

    metadata = engineer.run_pipeline(
        n_series=n_series,
        store_ids=store_ids,
        cat_ids=cat_ids,
        dept_ids=dept_ids,
        selection_strategy=args.selection_strategy,
        random_state=args.random_state,
    )

    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Series Selected:     {metadata['dataset_info']['total_series_selected']}")
    print(f"Historical Days Preserved: {metadata['dataset_info']['total_days_historical']} days per series")
    print(f"Stores Included:           {', '.join(metadata['dataset_info']['selected_stores'])}")
    print(f"Categories Included:       {', '.join(metadata['dataset_info']['selected_categories'])}")
    print(f"Feature Count:             {metadata['feature_engineering']['total_feature_count']}")
    print(f"Features:                  {', '.join(metadata['feature_engineering']['all_feature_columns'])}")
    print("-" * 70)
    print(f"Validation Horizon:        {metadata['split_info']['validation_horizon_days']} days")
    print(f"Train Period:              {metadata['split_info']['train_start_date']} to {metadata['split_info']['train_end_date']}")
    print(f"Validation Period:         {metadata['split_info']['val_start_date']} to {metadata['split_info']['val_end_date']}")
    print(f"Train Shape:               {metadata['data_shapes']['train_shape']}")
    print(f"Validation Shape:          {metadata['data_shapes']['val_shape']}")
    print("-" * 70)
    print("Artifacts Saved:")
    for name, path in metadata["saved_files"].items():
        print(f"  - {name:<18} -> {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
