#!/usr/bin/env python
"""
Post-race pipeline: update caches with a completed race, then train a
round-specific prediction model for the next race.

Usage
-----
    # After Australian GP 2026 (Round 1) finishes:
    python scripts/post_race.py --year 2026 --completed-round 1

    # Equivalent using event name:
    python scripts/post_race.py --year 2026 --completed-event "Australian Grand Prix"

    # Force retrain even if model already exists:
    python scripts/post_race.py --year 2026 --completed-round 1 --force

Pipeline
--------
    1. Fetch completed race results → append to raw_results cache
    2. Delete derived caches (features, practice, qualifying)
       → they are rebuilt automatically during training from local FastF1 session files
    3. Train ranker_{year}_R{next_round:02d}.pkl  (the model for the upcoming race)
    4. Save training metadata sidecar JSON

After this script runs, the live predictor will automatically use the new
round-specific model when predicting the upcoming race.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1
import pandas as pd

from src.config import (
    CACHE_DIR,
    DEFAULT_MODE,
    DEFAULT_RANKER_PARAMS,
    DEFAULT_REGRESSOR_PARAMS,
    FEATURES_CACHE,
    MODELS_DIR,
    PRACTICE_CACHE,
    QUALIFYING_CACHE,
    RAW_RESULTS_CACHE,
)
from src.data.loaders import build_training_dataset
from src.models.ranker import RaceRanker
from src.utils.helpers import get_drop_columns, get_race_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _append_race_results(year: int, event_name: str) -> bool:
    """Fetch race results from FastF1 and append to RAW_RESULTS_CACHE.

    Returns True if new data was added, False if already in cache.
    """
    raw = pd.read_pickle(RAW_RESULTS_CACHE) if RAW_RESULTS_CACHE.exists() else pd.DataFrame()

    if not raw.empty:
        if not raw[(raw["Year"] == year) & (raw["EventName"] == event_name)].empty:
            log.info("  Race results already in cache: %d / %s", year, event_name)
            return False

    log.info("  Fetching race results from FastF1: %d / %s …", year, event_name)
    session = fastf1.get_session(year, event_name, "Race")
    session.load()
    results = session.results.copy()
    if results is None or len(results) == 0:
        log.warning("  No race results returned for %d / %s", year, event_name)
        return False

    event_data = session.event
    results["Year"]              = year
    results["RoundNumber"]       = event_data["RoundNumber"]
    results["EventName"]         = event_data["EventName"]
    results["Location"]          = event_data["Location"]
    results["Country"]           = event_data["Country"]
    results["EventDate"]         = event_data["EventDate"]
    results["EventFormat"]       = event_data["EventFormat"]
    results["OfficialEventName"] = event_data["OfficialEventName"]
    for i in range(1, 6):
        if event_data.get(f"Session{i}") == "Race":
            results["RaceDateTime"] = event_data.get(f"Session{i}Date")
            break

    updated = pd.concat([raw, results], ignore_index=True)
    RAW_RESULTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    updated.to_pickle(RAW_RESULTS_CACHE)
    log.info("  Appended %d rows → raw results cache (%d total)", len(results), len(updated))
    return True


def refresh_caches_for_race(year: int, event_name: str) -> None:
    """Append a completed race to the raw results cache and wipe derived caches.

    The derived caches (features, practice, qualifying) are deleted so they
    will be fully rebuilt from the local FastF1 session files on the next
    training run — no separate per-cache append logic needed.
    """
    fastf1.Cache.offline_mode = False

    log.info("━━━ Refreshing caches for %d — %s ━━━", year, event_name)

    _append_race_results(year, event_name)

    for cache_path, label in [
        (FEATURES_CACHE,   "features cache"),
        (PRACTICE_CACHE,   "practice cache"),
        (QUALIFYING_CACHE, "qualifying cache"),
    ]:
        if cache_path.exists():
            cache_path.unlink()
            log.info("  Deleted %s → will rebuild from FastF1 session files", label)

    log.info("Cache refresh complete. Derived caches will rebuild on next training run.")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _save_meta(model_name: str, test_year: int, test_round: int, train_data: pd.DataFrame) -> None:
    """Save a JSON metadata sidecar next to the model pickle."""
    meta: dict = {
        "test_year":       test_year,
        "test_round":      test_round,
        "n_training_rows": int(len(train_data)),
        "trained_at":      datetime.now(timezone.utc).isoformat(),
    }

    if len(train_data):
        last = train_data.sort_values("EventDate").iloc[-1]
        meta["training_cutoff"] = {
            "year":  int(last["Year"]),
            "round": int(last["RoundNumber"]),
            "event": str(last["EventName"]),
            "date":  str(pd.to_datetime(last["EventDate"]).date()),
        }
    else:
        meta["training_cutoff"] = None

    meta_path = MODELS_DIR / model_name.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info("Metadata saved → %s", meta_path)


def train_for_round(year: int, next_round: int, mode: str = DEFAULT_MODE) -> None:
    """Train and save ranker_{year}_R{next_round:02d}.pkl.

    Training data = all races strictly before *next_round* in *year*
    (i.e. 2018–(year-1) + any completed rounds in *year* before next_round).
    Derived caches (practice, qualifying, features) are rebuilt automatically
    by build_training_dataset from the local FastF1 session files.
    """
    model_name = f"ranker_{year}_R{next_round:02d}.pkl"
    log.info("━━━ Training %s (predicts round %d of %d) ━━━",
             model_name, next_round, year)

    train_data, test_data = build_training_dataset(
        test_year     = year,
        test_round    = next_round,
        force_rebuild = False,
    )

    if train_data.empty:
        log.error("No training data available — aborting.")
        return

    drop_cols = get_drop_columns(train_data)
    X_train   = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    y_train   = train_data["Position"].astype(float)

    log.info("Training: %d rows, %d features", len(X_train), X_train.shape[1])

    if not test_data.empty:
        X_test = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
        X_test = X_test.reindex(columns=X_train.columns)
        y_test = test_data["Position"].astype(float)
    else:
        X_test, y_test = None, None
        log.info("Round %d not in cache yet — skipping evaluation", next_round)

    default_params = DEFAULT_REGRESSOR_PARAMS if mode == "regressor" else DEFAULT_RANKER_PARAMS
    ranker = RaceRanker(params=default_params.copy(), mode=mode)

    if mode == "ranker":
        ranker.fit(X_train, y_train, get_race_groups(train_data))
    else:
        ranker.fit(X_train, y_train)

    if X_test is not None and len(X_test):
        mae = ranker.evaluate(X_test, y_test, test_data["EventName"])
        log.info("Hold-out MAE (round %d): %.3f positions", next_round, mae)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ranker.save(MODELS_DIR / model_name)
    log.info("Model saved → %s", MODELS_DIR / model_name)

    _save_meta(model_name, test_year=year, test_round=next_round, train_data=train_data)

    if len(train_data):
        cutoff = train_data.sort_values("EventDate").iloc[-1]
        log.info(
            "Training cutoff: %d %s (Round %d)",
            int(cutoff["Year"]), cutoff["EventName"], int(cutoff["RoundNumber"]),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_event_name(year: int, round_number: int) -> str:
    """Look up event name for a given year/round from raw results cache or FastF1."""
    if RAW_RESULTS_CACHE.exists():
        raw = pd.read_pickle(RAW_RESULTS_CACHE)
        row = raw[(raw["Year"] == year) & (raw["RoundNumber"] == round_number)]
        if not row.empty:
            return str(row.iloc[0]["EventName"])

    fastf1.Cache.offline_mode = False
    schedule = fastf1.get_event_schedule(year)
    row = schedule[schedule["RoundNumber"] == round_number]
    if not row.empty:
        return str(row.iloc[0]["EventName"])

    raise ValueError(f"Could not resolve event name for {year} round {round_number}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-race pipeline: update caches and train next-race model."
    )
    p.add_argument("--year", type=int, default=datetime.now().year,
                   help="Season year (default: current year).")

    event_group = p.add_mutually_exclusive_group(required=True)
    event_group.add_argument("--completed-round", type=int, metavar="N",
                              help="Round number of the race that just finished.")
    event_group.add_argument("--completed-event", type=str, metavar="NAME",
                              help="Event name of the race that just finished.")

    p.add_argument("--next-round", type=int, default=None,
                   help="Round number to train a model for (default: completed-round + 1).")
    p.add_argument("--mode", type=str, default=DEFAULT_MODE,
                   choices=["regressor", "ranker"],
                   help="Model type (default: regressor).")
    p.add_argument("--skip-cache-refresh", action="store_true",
                   help="Skip cache refresh and only (re)train the model.")
    p.add_argument("--force", action="store_true",
                   help="Retrain model even if it already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Resolve event name and round number
    if args.completed_round is not None:
        completed_round = args.completed_round
        completed_event = _resolve_event_name(args.year, completed_round)
    else:
        completed_event = args.completed_event
        if RAW_RESULTS_CACHE.exists():
            raw = pd.read_pickle(RAW_RESULTS_CACHE)
            row = raw[(raw["Year"] == args.year) & (raw["EventName"] == completed_event)]
            completed_round = int(row.iloc[0]["RoundNumber"]) if not row.empty else None
        else:
            completed_round = None

        if completed_round is None:
            fastf1.Cache.offline_mode = False
            schedule = fastf1.get_event_schedule(args.year)
            row = schedule[schedule["EventName"] == completed_event]
            if row.empty:
                log.error("Event '%s' not found in %d schedule.", completed_event, args.year)
                sys.exit(1)
            completed_round = int(row.iloc[0]["RoundNumber"])

    next_round = args.next_round if args.next_round is not None else completed_round + 1

    log.info("Completed race : %d — %s (Round %d)", args.year, completed_event, completed_round)
    log.info("Training model for: Round %d", next_round)

    # ── Phase 1: Update caches ────────────────────────────────────────────
    if not args.skip_cache_refresh:
        refresh_caches_for_race(args.year, completed_event)
    else:
        log.info("Skipping cache refresh (--skip-cache-refresh)")

    # ── Phase 2: Train model ──────────────────────────────────────────────
    model_name = f"ranker_{args.year}_R{next_round:02d}.pkl"
    if (MODELS_DIR / model_name).exists() and not args.force:
        log.info("Model already exists: %s (use --force to retrain)", model_name)
    else:
        train_for_round(args.year, next_round, mode=args.mode)

    log.info(
        "Post-race pipeline complete.\n"
        "  Completed: %d %s (Round %d)\n"
        "  Model ready: models/%s",
        args.year, completed_event, completed_round, model_name,
    )


if __name__ == "__main__":
    main()
