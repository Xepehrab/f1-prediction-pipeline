"""Root-level CLI shim — delegates entirely to the package CLI.

This replaces the original broken version that imported non-existent functions
(run_ingestion, run_cleaning, build_features).

Usage examples:
    python cli.py build
    python cli.py train --target podium
    python cli.py predict --season 2026 --round 5 --target podium
"""

from f1_prediction.cli import main

if __name__ == "__main__":
    main()