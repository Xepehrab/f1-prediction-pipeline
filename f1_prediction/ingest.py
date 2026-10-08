"""Download race, qualifying, and sprint tables from the Jolpica (Ergast) API."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from .config import (
    INTERIM_DIR,
    JOLPICA_BASE,
    RAW_DIR,
    REQUEST_PAUSE_S,
    USER_AGENT,
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})


def _cache_path(resource: str, year: int, offset: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR / f"{resource}_{year}_offset_{offset}.json"


def _get_json(url: str, dest: Path) -> dict:
    if dest.exists():
        return json.loads(dest.read_text(encoding="utf-8"))
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = SESSION.get(url, timeout=45)
            if response.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            dest.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(REQUEST_PAUSE_S)
            return payload
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def _paginate_year(resource: str, year: int) -> list[dict]:
    """Paginate `/ {year}/{resource}.json` and return Race objects (merged)."""
    races: dict[tuple[str, str], dict] = {}
    offset = 0
    limit = 100
    total = None
    while total is None or offset < total:
        url = f"{JOLPICA_BASE}/{year}/{resource}.json?limit={limit}&offset={offset}"
        payload = _get_json(url, _cache_path(resource, year, offset))
        mr = payload["MRData"]
        total = int(mr["total"])
        table_key = "RaceTable"
        for race in mr.get(table_key, {}).get("Races", []):
            key = (str(race["season"]), str(race["round"]))
            if key not in races:
                races[key] = race
            else:
                for list_name in ("Results", "QualifyingResults", "SprintResults"):
                    extra = race.get(list_name)
                    if extra:
                        races[key].setdefault(list_name, []).extend(extra)
        offset += limit
        if total == 0:
            break
    return list(races.values())


def _lap_to_seconds(value: str | None) -> float | None:
    if not value or value in {"", "\\N"}:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def parse_results(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        circuit = race["Circuit"]
        loc = circuit["Location"]
        for res in race.get("Results", []):
            driver = res["Driver"]
            constructor = res["Constructor"]
            fastest = res.get("FastestLap") or {}
            fastest_time = (fastest.get("Time") or {}).get("time")
            avg_speed = (fastest.get("AverageSpeed") or {}).get("speed")
            race_time = (res.get("Time") or {}).get("millis")
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "race_date": race.get("date"),
                    "circuit_id": circuit["circuitId"],
                    "circuit_name": circuit["circuitName"],
                    "country": loc.get("country"),
                    "locality": loc.get("locality"),
                    "lat": float(loc["lat"]) if loc.get("lat") else None,
                    "lng": float(loc["long"]) if loc.get("long") else None,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code"),
                    "driver_number": res.get("number"),
                    "given_name": driver.get("givenName"),
                    "family_name": driver.get("familyName"),
                    "date_of_birth": driver.get("dateOfBirth"),
                    "nationality": driver.get("nationality"),
                    "constructor_id": constructor["constructorId"],
                    "constructor_name": constructor["name"],
                    "grid": int(res.get("grid") or 0),
                    "finish_position": int(res["position"]),
                    "position_text": res.get("positionText"),
                    "points": float(res.get("points") or 0),
                    "laps": int(res.get("laps") or 0),
                    "status": res.get("status"),
                    "race_time_ms": int(race_time) if race_time else None,
                    "fastest_lap_rank": int(fastest["rank"]) if fastest.get("rank") else None,
                    "fastest_lap_time_s": _lap_to_seconds(fastest_time),
                    "fastest_lap_avg_speed": float(avg_speed) if avg_speed else None,
                }
            )
    return pd.DataFrame(rows)


def parse_qualifying(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for res in race.get("QualifyingResults", []):
            driver = res["Driver"]
            constructor = res["Constructor"]
            q1 = _lap_to_seconds(res.get("Q1"))
            q2 = _lap_to_seconds(res.get("Q2"))
            q3 = _lap_to_seconds(res.get("Q3"))
            candidates = [t for t in (q1, q2, q3) if t is not None]
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": driver["driverId"],
                    "constructor_id": constructor["constructorId"],
                    "quali_position": int(res["position"]),
                    "q1_s": q1,
                    "q2_s": q2,
                    "q3_s": q3,
                    "best_quali_s": min(candidates) if candidates else None,
                }
            )
    return pd.DataFrame(rows)


def parse_sprint(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for res in race.get("SprintResults", []):
            driver = res["Driver"]
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": driver["driverId"],
                    "sprint_grid": int(res.get("grid") or 0),
                    "sprint_position": int(res["position"]),
                    "sprint_points": float(res.get("points") or 0),
                    "sprint_status": res.get("status"),
                }
            )
    return pd.DataFrame(rows)


def ingest(start_year: int, end_year: int) -> dict[str, pd.DataFrame]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    result_frames = []
    quali_frames = []
    sprint_frames = []
    for year in range(start_year, end_year + 1):
        print(f"Ingesting {year}...")
        results_year = parse_results(_paginate_year("results", year))
        if not results_year.empty:
            result_frames.append(results_year)
        quali_year = parse_qualifying(_paginate_year("qualifying", year))
        if not quali_year.empty:
            quali_frames.append(quali_year)
        sprint_df = parse_sprint(_paginate_year("sprint", year))
        if not sprint_df.empty:
            sprint_frames.append(sprint_df)

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    qualifying = pd.concat(quali_frames, ignore_index=True) if quali_frames else pd.DataFrame()
    sprint = pd.concat(sprint_frames, ignore_index=True) if sprint_frames else pd.DataFrame()

    results.to_csv(INTERIM_DIR / "results.csv", index=False)
    qualifying.to_csv(INTERIM_DIR / "qualifying.csv", index=False)
    sprint.to_csv(INTERIM_DIR / "sprint.csv", index=False)
    print(
        f"Saved interim tables: results={len(results)}, "
        f"qualifying={len(qualifying)}, sprint={len(sprint)}"
    )
    return {"results": results, "qualifying": qualifying, "sprint": sprint}


def load_interim() -> dict[str, pd.DataFrame]:
    return {
        "results": pd.read_csv(INTERIM_DIR / "results.csv"),
        "qualifying": pd.read_csv(INTERIM_DIR / "qualifying.csv"),
        "sprint": pd.read_csv(INTERIM_DIR / "sprint.csv")
        if (INTERIM_DIR / "sprint.csv").exists()
        else pd.DataFrame(),
    }
