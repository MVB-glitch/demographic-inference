# -*- coding: utf-8 -*-
"""
CLI interface for the Demographic Inference Pipeline.

Usage:
    demoinfer --csv data.csv --images ./images/
    demoinfer --csv data.csv --images ./images/ --limit 10 --batch-size 5
    python -m demographic_inference --csv data.csv --images ./images/
"""

import argparse
import sys

from .config import PipelineConfig, resolve_api_key
from .data_loader import load_users_from_csv
from .metrics import export_results, generate_summary_dataframe, print_metrics
from .pipeline import run_pipeline


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="demoinfer",
        description=(
            "Hybrid LLM + Dempster-Shafer Theory pipeline for inferring "
            "demographics (age, gender, location) from X (Twitter) user data."
        ),
    )

    parser.add_argument(
        "--csv", required=True,
        help="Path to input CSV file with user data.",
    )
    parser.add_argument(
        "--images", default=None,
        help="Path to directory containing user images.",
    )
    parser.add_argument(
        "--output", default="results.xlsx",
        help="Output file path (default: results.xlsx).",
    )
    parser.add_argument(
        "--format", default="xlsx", choices=["xlsx", "csv", "json"],
        help="Output format (default: xlsx).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N users.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of users per batch for checkpointing (default: 5).",
    )
    parser.add_argument(
        "--max-posts", type=int, default=15,
        help="Maximum number of posts to process per user (default: 15).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash",
        help="Gemini model name (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints/).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh, ignoring any existing checkpoint.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Main CLI entry point."""
    args = parse_args(argv)

    # Resolve API key
    api_key = resolve_api_key(args.api_key)

    # Build configuration
    config = PipelineConfig(
        api_key=api_key,
        model_name=args.model,
        csv_path=args.csv,
        image_dir=args.images,
        output_path=args.output,
        output_format=args.format,
        max_posts=args.max_posts,
        user_limit=args.limit,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose,
    )

    # Load data
    print(f"📂 Loading users from: {args.csv}")
    users = load_users_from_csv(args.csv, config)
    print(f"✅ Loaded {len(users)} users.\n")

    if not users:
        print("❌ No users found in CSV. Exiting.")
        sys.exit(1)

    # Run pipeline
    results = run_pipeline(
        users=users,
        config=config,
        resume=not args.no_resume,
    )

    # Print metrics
    print_metrics(results, config.kanto_prefectures)

    # Display summary table
    df_summary = generate_summary_dataframe(results)
    print(f"\n{df_summary.to_string(index=False)}")

    # Export
    export_results(results, config.output_path, config.output_format)


if __name__ == "__main__":
    main()
