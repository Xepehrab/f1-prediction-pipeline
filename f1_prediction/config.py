"""Shared paths, API settings, year range, and constructor lineage."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1-prediction-pipeline/0.1 (local research)"
REQUEST_PAUSE_S = 0.35
DEFAULT_START_YEAR = 2018
DEFAULT_END_YEAR = 2026
CONSTRUCTOR_LINEAGE = {
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    "toro_rosso": "rb",
    "alphatauri": "rb",
    "rb": "rb",
    "alfa": "audi",
    "sauber": "audi",
    "audi": "audi",
    "renault": "alpine",
    "alpine": "alpine",
    "red_bull": "red_bull",
    "mercedes": "mercedes",
    "ferrari": "ferrari",
    "mclaren": "mclaren",
    "williams": "williams",
    "haas": "haas",
    "cadillac": "cadillac",
}


def regulation_era(year: int) -> str:
    if year >= 2026:
        return "2026_new_regs"
    if year >= 2022:
        return "2022_2025_ground_effect"
    return "2018_2021_hybrid"
