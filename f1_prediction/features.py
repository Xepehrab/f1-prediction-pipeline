"""Leakage-safe features: anything from race N uses only data from races < N, plus pre-race weekend info."""

from __future__ import annotations

import pandas as pd

from .config import PROCESSED_DIR
from .tracks import track_row

TARGET_COLUMNS = [
    "finish_position",
    "won",
    "podium",
    "points_finish",
    "is_dnf",
    "points",
    "is_classified",
]

# Known before lights out (qualifying, sprint, grid, static track, lagged form).
PRE_RACE_COLUMNS = [
    "grid",
    "started_from_pit",
    "quali_position",
    "best_quali_s",
    "quali_gap_to_pole_s",
    "q3_s",
    "grid_vs_quali",
    "teammate_quali_gap_s",
    "beat_teammate_quali",
    "is_sprint_weekend",
    "sprint_position",
    "sprint_points",
    "driver_age_years",
    "field_size",
    "round",
]


def _lagged_rolling(
    df: pd.DataFrame,
    group: str,
    source: str,
    windows: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    grouped = df.groupby(group, sort=False)[source]
    shifted = grouped.shift(1)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_last"] = shifted
    for window in windows:
        # Capture window via default arg to avoid late-binding closure bug.
        out[f"{prefix}_mean_{window}"] = shifted.groupby(df[group]).transform(
            lambda s, w=window: s.rolling(w, min_periods=1).mean()
        )
        if source in {"is_dnf", "won", "podium", "points_finish"}:
            continue
        out[f"{prefix}_std_{window}"] = shifted.groupby(df[group]).transform(
            lambda s, w=window: s.rolling(w, min_periods=2).std()
        )
    return out


def engineer_features(driver_race: pd.DataFrame) -> pd.DataFrame:
    df = driver_race.sort_values(["race_date", "season", "round"]).copy()
    df["race_id"] = df["season"].astype(str) + "_" + df["round"].astype(str).str.zfill(2)

    track_df = pd.DataFrame(df["circuit_id"].map(track_row).tolist(), index=df.index)
    df = pd.concat([df, track_df], axis=1)
    # LightGBM needs categorical dtype for string columns; cast track_type now.
    df["track_type"] = df["track_type"].astype("category")

    df = pd.concat(
        [
            df,
            _lagged_rolling(df, "driver_id", "finish_position", (3, 5, 10), "driver_finish"),
            _lagged_rolling(df, "driver_id", "points", (3, 5, 10), "driver_points"),
            _lagged_rolling(df, "driver_id", "grid", (5,), "driver_grid"),
            _lagged_rolling(df, "driver_id", "is_dnf", (10,), "driver_dnf"),
            _lagged_rolling(df, "driver_id", "won", (20,), "driver_win_rate"),
            _lagged_rolling(df, "driver_id", "podium", (10,), "driver_podium"),
            _lagged_rolling(df, "constructor_lineage", "finish_position", (3, 5, 10), "team_finish"),
            _lagged_rolling(df, "constructor_lineage", "points", (3, 5, 10), "team_points"),
            _lagged_rolling(df, "constructor_lineage", "is_dnf", (10,), "team_dnf"),
        ],
        axis=1,
    )

    df["driver_circuit_key"] = df["driver_id"] + "|" + df["circuit_id"]
    df["team_circuit_key"] = df["constructor_lineage"] + "|" + df["circuit_id"]
    df["driver_era_key"] = df["driver_id"] + "|" + df["era"]
    df["team_era_key"] = df["constructor_lineage"] + "|" + df["era"]

    df = pd.concat(
        [
            df,
            _lagged_rolling(df, "driver_circuit_key", "finish_position", (4,), "driver_circuit_finish"),
            _lagged_rolling(df, "team_circuit_key", "finish_position", (4,), "team_circuit_finish"),
            _lagged_rolling(df, "driver_era_key", "finish_position", (5,), "driver_era_finish"),
            _lagged_rolling(df, "team_era_key", "finish_position", (5,), "team_era_finish"),
            _lagged_rolling(df, "driver_era_key", "points", (5,), "driver_era_points"),
            _lagged_rolling(df, "team_era_key", "points", (5,), "team_era_points"),
        ],
        axis=1,
    )

    prev_team = df.groupby("driver_id")["constructor_lineage"].shift(1)
    df["new_team"] = (
        prev_team.notna() & (df["constructor_lineage"] != prev_team)
    ).astype(int)

    season_points = df.groupby(["season", "driver_id"], sort=False)["points"].shift(1)
    df["season_points_before"] = (
        season_points.groupby([df["season"], df["driver_id"]]).cumsum().fillna(0)
    )
    team_season_points = df.groupby(["season", "constructor_lineage"], sort=False)["points"].shift(1)
    df["team_season_points_before"] = (
        team_season_points.groupby([df["season"], df["constructor_lineage"]]).cumsum().fillna(0)
    )

    df["grid_over_field"] = df["grid"] / df["field_size"]
    df["quali_over_field"] = df["quali_position"] / df["field_size"]
    df["sprint_position"] = df["sprint_position"].fillna(-1)
    df["sprint_points"] = df["sprint_points"].fillna(0)
    df["sprint_grid"] = df["sprint_grid"].fillna(-1) if "sprint_grid" in df.columns else -1

    feature_cols = [
        col
        for col in df.columns
        if col.startswith(
            (
                "driver_finish_",
                "driver_points_",
                "driver_grid_",
                "driver_dnf_",
                "driver_win_rate_",
                "driver_podium_",
                "team_finish_",
                "team_points_",
                "team_dnf_",
                "driver_circuit_",
                "team_circuit_",
                "driver_era_",
                "team_era_",
                "track_",
            )
        )
        or col
        in {
            "new_team",
            "season_points_before",
            "team_season_points_before",
            "grid_over_field",
            "quali_over_field",
            "era",
            "circuit_id",
            "constructor_lineage",
            "constructor_id",
            "driver_id",
            "season",
            "race_id",
            "race_date",
            "race_name",
            *PRE_RACE_COLUMNS,
        }
    ]

    keep = list(dict.fromkeys(["race_id", "season", "round", "race_date", "race_name", "driver_id"] + feature_cols + TARGET_COLUMNS))
    features = df[keep].copy()

    dictionary = pd.DataFrame(
        {
            "column": features.columns,
            "role": [
                "id"
                if col in {"race_id", "season", "round", "race_date", "race_name", "driver_id", "constructor_id", "constructor_lineage", "circuit_id", "era"}
                else "target"
                if col in TARGET_COLUMNS
                else "feature"
                for col in features.columns
            ],
            "leakage_safe": [
                "n/a"
                if col in TARGET_COLUMNS or col in {"race_id", "season", "round", "race_date", "race_name"}
                else "yes"
                for col in features.columns
            ],
        }
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "features.csv"
    dict_path = PROCESSED_DIR / "feature_dictionary.csv"
    features.to_csv(features_path, index=False)
    dictionary.to_csv(dict_path, index=False)

    qa = _qa_report(features)
    qa_path = PROCESSED_DIR / "qa_report.csv"
    qa.to_csv(qa_path, index=False)
    print(f"Feature table: {features.shape[0]} rows x {features.shape[1]} cols -> {features_path}")
    print(qa.to_string(index=False))
    return features


def _qa_report(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "n_rows", "value": len(features)})
    rows.append({"metric": "n_races", "value": features["race_id"].nunique()})
    rows.append({"metric": "n_drivers", "value": features["driver_id"].nunique()})
    rows.append({"metric": "min_date", "value": str(features["race_date"].min())[:10]})
    rows.append({"metric": "max_date", "value": str(features["race_date"].max())[:10]})
    rows.append({"metric": "missing_grid_pct", "value": float(features["grid"].isna().mean())})
    rows.append(
        {
            "metric": "missing_quali_pct",
            "value": float(features["quali_position"].isna().mean()),
        }
    )
    rows.append({"metric": "dnf_rate", "value": float(features["is_dnf"].mean())})
    rows.append(
        {
            "metric": "missing_form_driver_finish_mean_5_pct",
            "value": float(features["driver_finish_mean_5"].isna().mean()),
        }
    )
    for year, count in features.groupby("season").size().items():
        rows.append({"metric": f"rows_{year}", "value": int(count)})
    return pd.DataFrame(rows)
