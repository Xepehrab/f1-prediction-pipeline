"""Join, standardize, and drop unusable rows from the raw Jolpica tables."""

from __future__ import annotations

import pandas as pd

from .config import CONSTRUCTOR_LINEAGE, PROCESSED_DIR, regulation_era


def _is_dnf(status: str) -> bool:
    text = str(status)
    if text in {"Finished", "Lapped"}:
        return False
    if text.startswith("+") and "Lap" in text:
        return False
    return True


def _classified(status: str, position_text: str) -> bool:
    if str(position_text) in {"R", "D", "W", "E", "F", "N"}:
        return False
    return not _is_dnf(status) or str(status).startswith("+")


def clean_driver_race(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = tables["results"].copy()
    qualifying = tables["qualifying"].copy()
    sprint = tables["sprint"].copy()

    results["race_date"] = pd.to_datetime(results["race_date"])
    results["date_of_birth"] = pd.to_datetime(results["date_of_birth"], errors="coerce")
    results["driver_age_years"] = (
        results["race_date"] - results["date_of_birth"]
    ).dt.days / 365.25
    results["constructor_lineage"] = (
        results["constructor_id"].map(CONSTRUCTOR_LINEAGE).fillna(results["constructor_id"])
    )
    results["era"] = results["season"].map(regulation_era)
    results["is_dnf"] = results["status"].map(_is_dnf).astype(int)
    results["is_classified"] = [
        int(_classified(status, pos))
        for status, pos in zip(results["status"], results["position_text"])
    ]
    results["finished_race"] = (~results["is_dnf"].astype(bool)).astype(int)
    # Ergast uses grid=0 for pit-lane / failed to set a time.
    n_starters = results.groupby(["season", "round"])["driver_id"].transform("count")
    results["grid_raw"] = results["grid"]
    results["started_from_pit"] = (results["grid"] <= 0).astype(int)
    results["grid"] = results["grid"].where(results["grid"] > 0, n_starters)
    results["won"] = (results["finish_position"] == 1).astype(int)
    results["podium"] = (results["finish_position"] <= 3).astype(int)
    results["points_finish"] = (results["finish_position"] <= 10).astype(int)

    quali_cols = [
        "season",
        "round",
        "driver_id",
        "quali_position",
        "q1_s",
        "q2_s",
        "q3_s",
        "best_quali_s",
    ]
    if not qualifying.empty:
        results = results.merge(qualifying[quali_cols], on=["season", "round", "driver_id"], how="left")
    else:
        for col in quali_cols[3:]:
            results[col] = pd.NA

    if not sprint.empty:
        sprint_cols = [
            "season",
            "round",
            "driver_id",
            "sprint_grid",
            "sprint_position",
            "sprint_points",
        ]
        results = results.merge(sprint[sprint_cols], on=["season", "round", "driver_id"], how="left")
        results["is_sprint_weekend"] = results["sprint_position"].notna().astype(int)
    else:
        results["sprint_grid"] = pd.NA
        results["sprint_position"] = pd.NA
        results["sprint_points"] = pd.NA
        results["is_sprint_weekend"] = 0

    pole = (
        results.loc[results["best_quali_s"].notna()]
        .groupby(["season", "round"], as_index=False)["best_quali_s"]
        .min()
        .rename(columns={"best_quali_s": "pole_time_s"})
    )
    results = results.merge(pole, on=["season", "round"], how="left")
    results["quali_gap_to_pole_s"] = results["best_quali_s"] - results["pole_time_s"]

    teammate = results[
        ["season", "round", "constructor_id", "driver_id", "best_quali_s", "quali_position", "grid"]
    ].copy()
    teammate = teammate.merge(
        teammate,
        on=["season", "round", "constructor_id"],
        suffixes=("", "_teammate"),
    )
    teammate = teammate[teammate["driver_id"] != teammate["driver_id_teammate"]]
    teammate = teammate.sort_values(["season", "round", "driver_id", "quali_position_teammate"])
    teammate = teammate.drop_duplicates(["season", "round", "driver_id"], keep="first")
    teammate["teammate_quali_gap_s"] = teammate["best_quali_s"] - teammate["best_quali_s_teammate"]
    teammate["beat_teammate_quali"] = (
        teammate["quali_position"] < teammate["quali_position_teammate"]
    ).astype(int)
    results = results.merge(
        teammate[
            [
                "season",
                "round",
                "driver_id",
                "driver_id_teammate",
                "teammate_quali_gap_s",
                "beat_teammate_quali",
            ]
        ],
        on=["season", "round", "driver_id"],
        how="left",
    )

    results["grid_vs_quali"] = results["grid"] - results["quali_position"]
    results["field_size"] = n_starters
    results = results.sort_values(["race_date", "season", "round", "finish_position"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "driver_race.csv"
    results.to_csv(out_path, index=False)
    print(f"Cleaned driver-race table: {len(results)} rows -> {out_path}")
    return results
