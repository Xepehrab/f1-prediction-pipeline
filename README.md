# F1 Race Prediction — Complete ML Pipeline

A leakage-safe Machine Learning pipeline for predicting Formula 1 race outcomes using historical data from the **Jolpica F1 API** (Ergast-compatible API).

The pipeline downloads historical race data, cleans and joins race results, qualifying, and sprint data, builds time-aware features without data leakage, and trains a **LightGBM** model for race outcome prediction.

### Pipeline

```text
Jolpica API
     │
     ▼
  Ingest
     │
     ▼
 data/raw/
     │
     ▼
  Clean
     │
     ▼
data/interim/
     │
     ▼
 Features
     │
     ▼
data/processed/
     │
     ▼
  Train
     │
     ▼
 data/models/
     │
     ▼
 Predict
```

## Project Structure

```text
f1_prediction/
│
├── f1_prediction/                 # Python package
│   ├── __init__.py
│   ├── __main__.py                # Entry point for: python -m f1_prediction
│   ├── cli.py                     # CLI commands
│   ├── config.py                  # Paths, API settings, year range, team lineage
│   ├── ingest.py                  # Jolpica API ingestion, pagination, and caching
│   ├── clean.py                   # Combines results, qualifying, and sprint data
│   ├── features.py                # Leakage-safe feature engineering
│   ├── tracks.py                  # Static circuit metadata
│   └── model.py                   # LightGBM training and prediction
│
├── data/
│   ├── raw/                       # Cached Jolpica API responses
│   │   ├── results_2018_offset_0.json
│   │   ├── qualifying_2018_offset_0.json
│   │   └── sprint_2018_offset_0.json
│   │
│   ├── interim/
│   │   ├── results.csv            # Race results
│   │   ├── qualifying.csv         # Q1/Q2/Q3 qualifying times
│   │   └── sprint.csv             # Sprint results
│   │
│   ├── processed/
│   │   ├── driver_race.csv        # Standardized driver-race table
│   │   ├── features.csv           # Model-ready feature matrix
│   │   ├── feature_dictionary.csv # Feature and target definitions
│   │   └── qa_report.csv          # Data quality report
│   │
│   └── models/
│       └── lgbm_{target}_model.pkl # Saved LightGBM model
│
├── requirements.txt
├── run_pipeline.py                # Shortcut for the CLI
└── README.md
```

## Features

The pipeline is designed around **time-aware, leakage-safe feature engineering**.

Historical driver and constructor statistics are calculated using only information available **before the target race**.

Examples include:

* Driver recent finishing-position statistics
* Driver recent form
* Constructor/team performance
* Season points accumulated before the race
* Circuit-specific historical performance
* Qualifying performance
* Sprint performance
* Starting grid position
* Driver age
* Circuit characteristics

Rolling historical statistics use `shift(1)` before calculating rolling values, ensuring that the current race result is never included in its own features.

## Data Source

Historical Formula 1 data is collected from the **Jolpica F1 API**, an Ergast-compatible API.

[Jolpica F1 API — GitHub](https://github.com/jolpica/jolpica-f1?utm_source=chatgpt.com)

The API responses are cached locally under:

```text
data/raw/
```

This allows the cleaning and feature-engineering stages to be re-run without making additional API requests.

## Requirements

* Python 3.10+
* pandas >= 2.0
* requests >= 2.31
* scikit-learn
* LightGBM

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Build the Complete Dataset

Download the default historical range, clean the data, and generate model-ready features:

```bash
python -m f1_prediction build
```

The default configuration uses:

```text
Start year: 2018
End year:   2026
```

### Build a Custom Year Range

```bash
python -m f1_prediction build --start-year 2020 --end-year 2025
```

### Download Data Only

To download and cache the API data without running the cleaning and feature-engineering stages:

```bash
python -m f1_prediction ingest
```

Custom range:

```bash
python -m f1_prediction ingest --start-year 2022 --end-year 2024
```

### Prepare Data from Cached Files

Re-run the cleaning and feature-engineering stages using the existing files in `data/raw/`.

No additional API requests are required:

```bash
python -m f1_prediction prepare
```

### Train the Model

Train the LightGBM model using the generated feature dataset:

```bash
python -m f1_prediction train
```

The default target is:

```text
podium
```

### Predict a Specific Race

For example, to generate predictions for the 2024 Bahrain Grand Prix:

```bash
python -m f1_prediction predict --season 2024 --round 1 --target podium
```

### Alternative Entry Point

The pipeline can also be executed through:

```bash
python run_pipeline.py build
```

## Outputs

| File                                    | Description                                               |
| --------------------------------------- | --------------------------------------------------------- |
| `data/interim/results.csv`              | Historical race results                                   |
| `data/interim/qualifying.csv`           | Q1, Q2, and Q3 qualifying data                            |
| `data/interim/sprint.csv`               | Sprint results for sprint weekends                        |
| `data/processed/driver_race.csv`        | Standardized driver-by-race dataset                       |
| `data/processed/features.csv`           | Model-ready feature matrix                                |
| `data/processed/feature_dictionary.csv` | Feature and target definitions                            |
| `data/processed/qa_report.csv`          | Data quality, row counts, missingness, and DNF statistics |
| `data/models/lgbm_{target}_model.pkl`   | Trained LightGBM model                                    |

## Leakage Prevention

Preventing data leakage is a core design requirement of this project.

### Historical Features

Historical statistics are calculated using only previous races.

For example:

```python
groupby(...)
    .shift(1)
    .rolling(...)
```

This prevents the result of the current race from being used to predict that same race.

### Allowed Pre-Race Information

The model may use information that is available before the race starts, including:

* Starting grid position
* Qualifying performance
* Sprint performance
* Driver age
* Historical driver performance
* Historical constructor performance
* Season statistics accumulated before the race
* Circuit metadata

### Information Not Available Before the Race

The following race-result information must **not** be used as features for predicting that race:

* Finishing position
* DNF status
* Race fastest lap
* Number of pit stops during the race
* In-race telemetry
* Any other information generated after the race has started

This separation ensures that the prediction task reflects a realistic **pre-race prediction scenario**.

## Prediction Targets

The generated `features.csv` contains several possible prediction targets:

| Target            | Type                  | Description                              |
| ----------------- | --------------------- | ---------------------------------------- |
| `finish_position` | Ranking               | Driver's finishing position              |
| `won`             | Binary classification | Whether the driver won the race          |
| `podium`          | Binary classification | Whether the driver finished in the top 3 |
| `points_finish`   | Binary classification | Whether the driver scored points         |
| `is_dnf`          | Binary classification | Whether the driver failed to finish      |
| `points`          | Regression            | Championship points scored in the race   |

The target can be selected when training or generating predictions.

## Constructor Lineage

Historical constructor names are normalized to account for team renames across seasons.

Examples include:

```text
Force India
      ↓
Racing Point
      ↓
Aston Martin
```

```text
Toro Rosso
      ↓
AlphaTauri
      ↓
RB
```

```text
Sauber
      ↓
Alfa Romeo
      ↓
Sauber
      ↓
Audi
```

```text
Renault
      ↓
Alpine
```

The normalization logic is maintained in:

```text
f1_prediction/config.py
```

## Circuit Metadata

Static circuit characteristics are maintained in:

```text
f1_prediction/tracks.py
```

The current metadata includes 1–5 scores for characteristics such as:

* `is_street`
* `overtaking_difficulty`
* `tyre_stress`
* `pu_stress`
* `safety_car_risk`

These features provide additional circuit-level context to the Machine Learning model.

## API Caching and Rate Limiting

Raw API responses are cached under:

```text
data/raw/
```

Cached files follow a structure similar to:

```text
results_2018_offset_0.json
qualifying_2018_offset_0.json
sprint_2018_offset_0.json
```

This means subsequent pipeline stages can reuse previously downloaded data without repeatedly querying the API.

The ingestion process uses a request pause of approximately:

```text
REQUEST_PAUSE_S = 0.35
```

which corresponds to roughly 4 requests per second.

## Data Quality

After each build, the pipeline generates:

```text
data/processed/qa_report.csv
```

The report includes checks such as:

* Row counts
* Missing values
* DNF rate
* Data availability
* Dataset structure

This provides a basic quality-control layer before the data is used for Machine Learning.

## Project Goal

The goal of this project is to build a reproducible Formula 1 Machine Learning pipeline that demonstrates:

* API-based data ingestion
* Data caching
* Data cleaning and integration
* Time-aware feature engineering
* Leakage prevention
* Historical driver and constructor statistics
* Circuit-level feature engineering
* Machine Learning model training
* Race-level prediction

The project is structured so that each stage of the pipeline can be run independently or as part of the complete workflow.
