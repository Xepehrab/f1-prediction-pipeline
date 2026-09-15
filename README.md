# F1 Race Prediction — Data Pipeline

Ingests official historical results from the [Jolpica / Ergast API](https://github.com/jolpica/jolpica-f1), cleans them into one row per driver per race, and builds **leakage-safe** features. This stops before any model training.

Pipeline: `Ingest (API → data/raw) → Clean (→ data/interim) → Features (→ data/processed)`

## Project Structure

## Project Structure

## Project Structure

```text
f1_prediction/
├── f1_prediction/              # Python package (pipeline code)
│   ├── __init__.py
│   ├── __main__.py             # python -m f1_prediction entry
│   ├── cli.py                  # CLI: ingest / prepare / build
│   ├── config.py               # Paths, API base, year range (2018-2026), lineage
│   ├── ingest.py               # Jolpica pagination + caching (0.35s pause, 4 req/s)
│   ├── clean.py                # join results+qualifying+sprint -> driver_race
│   ├── features.py             # rolling features with shift(1) (no leakage)
│   └── tracks.py               # static circuit meta (is_street, tyre_stress...)
├── data/
│   ├── raw/                    # cached Jolpica JSON pages (per-year, per-offset)
│   │   ├── results_2018_offset_0.json ...
│   │   ├── qualifying_2018_offset_0.json ...
│   │   └── sprint_2018_offset_0.json ...
│   ├── interim/
│   │   ├── results.csv         # raw race results
│   │   ├── qualifying.csv      # Q1/Q2/Q3 times
│   │   └── sprint.csv          # sprint results (when available)
│   └── processed/
│       ├── driver_race.csv         # joined, standardized table (1 row = driver x race)
│       ├── features.csv            # model-ready matrix
│       ├── feature_dictionary.csv  # id / feature / target roles
│       └── qa_report.csv           # row counts, missingness, DNF rate
├── requirements.txt            # pandas>=2.0, requests>=2.31
├── run_pipeline.py             # shortcut -> f1_prediction.cli:main
└── README.md
```


## Setup
bash
cd f1_prediction
pip install -r requirements.txt

Requirements: `pandas>=2.0`, `requests>=2.31`, Python 3.10+

## Run

bash
download 2018–2026, clean, write features (full build)
python -m f1_prediction build

custom year range
python -m f1_prediction build --start-year 2020 --end-year 2025

re-run clean + features from cached JSON (no API calls)
python -m f1_prediction prepare

only download (no cleaning)
python -m f1_prediction ingest
python -m f1_prediction ingest --start-year 2022 --end-year 2024

alternative entry
python run_pipeline.py build (1/2)


Raw API pages are cached under `data/raw/` so later runs do not hammer the API (limit ~4 req/s, 500/hour, `REQUEST_PAUSE_S=0.35`).

## Outputs

| File | What it is |
|---|---|
| `data/interim/results.csv` | Race results |
| `data/interim/qualifying.csv` | Q1/Q2/Q3 times |
| `data/interim/sprint.csv` | Sprint results (when the weekend had one) |
| `data/processed/driver_race.csv` | Joined, standardized driver-race table |
| `data/processed/features.csv` | Model-ready matrix |
| `data/processed/feature_dictionary.csv` | `id` / `feature` / `target` roles |
| `data/processed/qa_report.csv` | Row counts, missingness, DNF rate |

## Leakage Rules

Form stats (`driver_finish_mean_5`, team points, circuit history, season points so far) use only **previous** races (`shift(1)` then rolling). Same-race inputs are only things known before lights out: grid, qualifying, sprint, track metadata, driver age.

Do **not** use as features for that race: finishing position, DNF, race fastest lap, race pit count, or in-race telemetry.

## Targets in `features.csv`

- `finish_position` — ranking label
- `won`, `podium`, `points_finish` — binary labels
- `is_dnf` — reliability label
- `points` — regression label

## Config (`f1_prediction/config.py`)

- `JOLPICA_BASE = https://api.jolpi.ca/ergast/f1`
- `DEFAULT_START_YEAR = 2018`, `DEFAULT_END_YEAR = 2026`
- `CONSTRUCTOR_LINEAGE` normalizes renames: force_india/racing_point→aston_martin, toro_rosso/alphatauri→rb, alfa/sauber→audi, renault→alpine
- `TRACK_META` in `tracks.py`: 1-5 scores per circuit (is_street, overtaking_difficulty, tyre_stress, pu_stress, safety_car_risk)

## Notes

- Sprint weekends have extra rows in `sprint.csv`; non-sprint weekends have none.
- DNF rate and missingness are summarized in `qa_report.csv` after each build.


