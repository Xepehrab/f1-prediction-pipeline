"""CLI: ingest Jolpica data, clean, engineer features, train, and predict."""

from __future__ import annotations

import argparse

from .clean import clean_driver_race
from .config import DEFAULT_END_YEAR, DEFAULT_START_YEAR
from .features import engineer_features
from .ingest import ingest, load_interim
from .model import predict_specific_race, train_baseline_model


def build(start_year: int, end_year: int, skip_ingest: bool = False) -> None:
    tables = load_interim() if skip_ingest else ingest(start_year, end_year)
    cleaned = clean_driver_race(tables)
    engineer_features(cleaned)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="F1 prediction pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ───────────────────────────────────────────────────────────────
    ingest_p = sub.add_parser("ingest", help="Download Jolpica results/qualifying/sprint")
    ingest_p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    ingest_p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)

    # ── prepare ──────────────────────────────────────────────────────────────
    sub.add_parser("prepare", help="Clean cached tables and write features (no API calls)")

    # ── build ─────────────────────────────────────────────────────────────────
    build_p = sub.add_parser("build", help="Ingest + clean + features")
    build_p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    build_p.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    build_p.add_argument("--skip-ingest", action="store_true")

    # ── train ─────────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train LightGBM model on the processed feature table")
    train_p.add_argument(
        "--target",
        default="podium",
        choices=["podium", "won", "points_finish", "is_dnf", "finish_position", "points"],
        help="Prediction target column (default: podium)",
    )
    train_p.add_argument(
        "--split-year",
        type=int,
        default=2025,
        help="First year used for the test split (default: 2025)",
    )

    # ── predict ───────────────────────────────────────────────────────────────
    predict_p = sub.add_parser("predict", help="Rank drivers for a specific race")
    predict_p.add_argument("--season", type=int, required=True, help="Season year, e.g. 2026")
    predict_p.add_argument("--round", type=int, required=True, dest="round_num", help="Round number, e.g. 5")
    predict_p.add_argument(
        "--target",
        default="podium",
        choices=["podium", "won", "points_finish", "is_dnf", "finish_position", "points"],
        help="Which trained model to use (default: podium)",
    )

    args = parser.parse_args(argv)

    if args.command == "ingest":
        ingest(args.start_year, args.end_year)
    elif args.command == "prepare":
        build(DEFAULT_START_YEAR, DEFAULT_END_YEAR, skip_ingest=True)
    elif args.command == "build":
        build(args.start_year, args.end_year, skip_ingest=args.skip_ingest)
    elif args.command == "train":
        train_baseline_model(target_col=args.target, split_year=args.split_year)
    elif args.command == "predict":
        predict_specific_race(
            season=args.season,
            round_num=args.round_num,
            target_col=args.target,
        )


if __name__ == "__main__":
    main()
