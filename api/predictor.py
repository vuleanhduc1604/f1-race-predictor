"""
Wrapper functions that expose the ML system as plain Python functions
returning dicts/lists — no printing, no CLI args.  Used by the FastAPI app.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

# Make sure project root is on the path when this module is imported directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CACHE_DIR, FEATURES_CACHE, MODELS_DIR, TEST_YEAR
from src.data.loaders import build_training_dataset
from src.models.ranker import RaceRanker
from src.utils.helpers import get_drop_columns

log = logging.getLogger(__name__)

# FastF1's HTTP cache needs a writable directory.
# Vercel's deployed /var/task filesystem is read-only; /tmp is always writable.
_FF1_HTTP_CACHE = "/tmp/fastf1_cache" if os.environ.get("VERCEL") else str(CACHE_DIR)

# Enable FastF1 cache once at import time (offline mode — no live API calls)
fastf1.Cache.enable_cache(_FF1_HTTP_CACHE)
fastf1.Cache.offline_mode = True


# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across requests
# ---------------------------------------------------------------------------

_ranker_cache: dict = {}          # keyed by year or "default"
_dataset_cache: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
_schedule_cache: dict[int, list[dict]] = {}  # keyed by year


def _load_ranker(year: int | None = None, round_number: int | None = None) -> RaceRanker:
    """
    Return the model for *year* / *round_number*.

    Resolution order:
      1. ranker_{year}_R{round:02d}.pkl  – round-specific (post-race pipeline)
      2. ranker_{year}.pkl               – year-specific
      3. ranker_2026.pkl                 – general 2026 model
      4. ranker.pkl                      – fallback
    """
    candidates: list[tuple[str, Path]] = []
    if year is not None and round_number is not None:
        name = f"ranker_{year}_R{round_number:02d}.pkl"
        candidates.append((name, MODELS_DIR / name))
    if year is not None:
        name = f"ranker_{year}.pkl"
        candidates.append((name, MODELS_DIR / name))
    candidates.append(("ranker_2026.pkl", MODELS_DIR / "ranker_2026.pkl"))
    candidates.append(("ranker.pkl",      MODELS_DIR / "ranker.pkl"))

    for cache_key, path in candidates:
        if path.exists():
            if cache_key not in _ranker_cache:
                log.info("Loading model from %s …", path)
                _ranker_cache[cache_key] = RaceRanker.load(path)
            return _ranker_cache[cache_key]

    raise FileNotFoundError(
        f"No model found for year={year}, round={round_number}. "
        "Run scripts/train.py first."
    )


def _load_model_meta(year: int | None = None, round_number: int | None = None) -> dict | None:
    """Load the training metadata JSON sidecar for the active model, or None if not found."""
    candidates: list[Path] = []
    if year is not None and round_number is not None:
        candidates.append(MODELS_DIR / f"ranker_{year}_R{round_number:02d}_meta.json")
    if year is not None:
        candidates.append(MODELS_DIR / f"ranker_{year}_meta.json")
    candidates.append(MODELS_DIR / "ranker_2026_meta.json")
    candidates.append(MODELS_DIR / "ranker_meta.json")

    for path in candidates:
        if path.exists():
            with open(path) as fh:
                return json.load(fh)
    return None


def _load_dataset(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if year not in _dataset_cache:
        log.info("Building dataset for year %d …", year)
        _dataset_cache[year] = build_training_dataset(test_year=year)
    return _dataset_cache[year]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_available_years() -> list[int]:
    """Return sorted list of years available for prediction.

    Includes years in the features cache (historical data) plus the current
    and next calendar year so live predictions for 2025/2026+ are accessible.
    """
    from datetime import date
    current_year = date.today().year

    if FEATURES_CACHE.exists():
        df = pd.read_pickle(FEATURES_CACHE)
        cache_years = set(int(y) for y in df["Year"].unique())
    else:
        cache_years = set(range(2018, TEST_YEAR + 1))

    # Always expose the current year for live predictions (next year stays hidden until it starts)
    live_years = {y for y in range(TEST_YEAR, current_year + 1) if y >= TEST_YEAR}
    return sorted(cache_years | live_years)


def _fetch_schedule_events(year: int) -> list[dict]:
    """
    Fetch the race calendar for *year* from FastF1 online and return the same
    dict shape as get_events().  Results are cached in _schedule_cache.
    """
    if year in _schedule_cache:
        return _schedule_cache[year]

    fastf1.Cache.offline_mode = False
    try:
        fastf1.Cache.enable_cache(_FF1_HTTP_CACHE)
        schedule = fastf1.get_event_schedule(year)
        race_events = schedule[schedule["EventFormat"] != "testing"]
        result = []
        for _, row in race_events.iterrows():
            result.append({
                "round":   int(row["RoundNumber"]),
                "name":    row["EventName"],
                "country": row.get("Country", ""),
                "date":    str(row["EventDate"])[:10] if "EventDate" in row else "",
            })
        _schedule_cache[year] = result
        return result
    except Exception as exc:
        log.warning("Could not fetch schedule for %d: %s", year, exc)
        return []
    finally:
        fastf1.Cache.offline_mode = True


def get_events(year: int) -> list[dict]:
    """
    Return events for *year* in race order.

    Each item: {"round": int, "name": str, "country": str, "date": str}

    For historical years (in FEATURES_CACHE) the data comes from the local
    pickle.  For future years (e.g. 2026) the F1 calendar is fetched online.
    """
    # Try local cache first
    if FEATURES_CACHE.exists():
        df = pd.read_pickle(FEATURES_CACHE)
        year_df = df[df["Year"] == year].copy()
        if not year_df.empty:
            cols = [c for c in ["RoundNumber", "EventName", "Country", "EventDate"] if c in year_df.columns]
            events = (
                year_df[cols]
                .drop_duplicates(subset=["EventName"])
                .sort_values("EventDate" if "EventDate" in cols else "RoundNumber")
            )
            result = []
            for _, row in events.iterrows():
                result.append({
                    "round":   int(row["RoundNumber"]) if "RoundNumber" in row else None,
                    "name":    row["EventName"],
                    "country": row.get("Country", ""),
                    "date":    str(row["EventDate"])[:10] if "EventDate" in row else "",
                })
            return result

    # Fall back to live schedule (for 2026+ or missing cache years)
    log.info("Year %d not in features cache — fetching schedule online …", year)
    return _fetch_schedule_events(year)


def run_prediction(year: int, event: str) -> dict:
    """
    Run predictions for a single event.

    Returns:
    {
      "year": int,
      "event": str,
      "has_actuals": bool,
      "mae": float | None,
      "drivers": [
        {"abbreviation": str, "grid_position": int,
         "predicted_position": int, "actual_position": int | None, "error": int | None},
        ...
      ]
    }
    """
    _, pred_data = _load_dataset(year)

    race = pred_data[pred_data["EventName"] == event].copy()
    if race.empty:
        raise ValueError(f"No data found for event '{event}' in {year}.")

    round_number = int(race["RoundNumber"].iloc[0]) if "RoundNumber" in race.columns else None
    ranker = _load_ranker(year, round_number=round_number)
    meta   = _load_model_meta(year, round_number=round_number)

    drop_cols = get_drop_columns(race)
    X = race.drop(columns=[c for c in drop_cols if c in race.columns])
    if ranker.feature_columns:
        X = X.reindex(columns=ranker.feature_columns)

    race["predicted_position"] = ranker.predict_positions(X, race["EventName"])

    has_actuals = bool("Position" in race.columns and race["Position"].notna().any())
    median_error = None
    if has_actuals:
        race["actual_position"] = race["Position"].astype(int)
        race["error"] = (race["predicted_position"] - race["actual_position"]).abs()
        median_error = float(race["error"].median())

    feature_cols = list(X.columns)

    def _serialize_val(v):
        if pd.isna(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return round(float(v), 4)
        return v

    drivers = []
    for _, row in race.sort_values("predicted_position").iterrows():
        x_row = X.loc[row.name]
        features = {col: _serialize_val(x_row[col]) for col in feature_cols}
        entry: dict = {
            "abbreviation": row.get("Abbreviation", ""),
            "full_name": f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip(),
            "team": row.get("TeamName", ""),
            "grid_position": int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None,
            "predicted_position": int(row["predicted_position"]),
            "actual_position": int(row["actual_position"]) if has_actuals else None,
            "error": int(row["error"]) if has_actuals else None,
            "features": features,
        }
        drivers.append(entry)

    return {
        "year": year,
        "event": event,
        "in_sample": year <= TEST_YEAR,
        "has_actuals": has_actuals,
        "median_error": round(median_error, 3) if median_error is not None else None,
        "feature_names": feature_cols,
        "training_cutoff": meta.get("training_cutoff") if meta else None,
        "drivers": drivers,
    }


def run_evaluation(year: int) -> dict:
    """
    Return full-season evaluation metrics for *year*.

    Returns:
    {
      "year": int,
      "median_error": float,
      "within_1": float,  # % of predictions within 1 position
      "within_2": float,
      "within_3": float,
      "within_5": float,
      "per_race": [{"event": str, "drivers": int, "median_error": float}, ...]
    }
    """
    ranker = _load_ranker(year)
    _, test_data = _load_dataset(year)

    if test_data.empty or "Position" not in test_data.columns:
        raise ValueError(f"No race results available for {year}. Season may not have started.")

    if test_data["Position"].isna().all():
        raise ValueError(f"No actual positions recorded for {year} yet.")

    drop_cols = get_drop_columns(test_data)
    X = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
    if ranker.feature_columns:
        X = X.reindex(columns=ranker.feature_columns)

    test_data = test_data.copy()
    test_data["predicted_position"] = ranker.predict_positions(X, test_data["EventName"])
    errors = (test_data["predicted_position"] - test_data["Position"]).abs()

    test_data["_err"] = errors
    per_race_df = (
        test_data.groupby("EventName")
        .agg(drivers=("Position", "count"), median_error=("_err", "median"))
        .reset_index()
        .sort_values("EventName")
    )
    per_race = [
        {
            "event": row["EventName"],
            "drivers": int(row["drivers"]),
            "median_error": round(float(row["median_error"]), 3),
        }
        for _, row in per_race_df.iterrows()
    ]

    return {
        "year": year,
        "median_error": round(float(errors.median()), 3),
        "within_1": round(float((errors <= 1).mean() * 100), 1),
        "within_2": round(float((errors <= 2).mean() * 100), 1),
        "within_3": round(float((errors <= 3).mean() * 100), 1),
        "within_5": round(float((errors <= 5).mean() * 100), 1),
        "per_race": per_race,
    }


def get_feature_importance(top_n: int = 25) -> list[dict]:
    """Return top-N feature importances as [{"feature": str, "importance": int}, ...]."""
    ranker = _load_ranker(None)
    imp = ranker.feature_importances.head(top_n)
    return [
        {"feature": row["Feature"], "importance": int(row["Importance"])}
        for _, row in imp.iterrows()
    ]
