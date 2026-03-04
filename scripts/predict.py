#!/usr/bin/env python
"""
Generate race-finish position predictions for an upcoming (or past) event.

Usage
-----
    # Predict a single race by year + event name
    python scripts/predict.py --year 2025 --event "Australian Grand Prix"

    # Predict all races in a season
    python scripts/predict.py --year 2025

    # Use a non-default model file
    python scripts/predict.py --year 2025 --event "Bahrain Grand Prix" \\
                              --model models/ranker_v2.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1
import pandas as pd

from src.config import CACHE_DIR, MODELS_DIR, TEST_YEAR
from src.data.loaders import build_training_dataset
from src.models.ranker import RaceRanker
from src.utils.helpers import get_drop_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict F1 race finishing positions.")
    p.add_argument("--year",  type=int, default=TEST_YEAR,
                   help="Season year to predict (default: TEST_YEAR).")
    p.add_argument("--event", type=str, default=None,
                   help="Event name to predict.  If omitted, predict all events in --year.")
    p.add_argument("--model", type=str, default=None,
                   help="Path to saved ranker .pkl file (default: models/ranker.pkl).")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of drivers to display per race (default: 20).")
    return p.parse_args()


def _display_race(
    race_df: pd.DataFrame,
    event_name: str,
    year: int,
    top_n: int,
) -> None:
    """Pretty-print predicted vs actual positions for one race."""
    cols = ["Abbreviation", "GridPosition", "predicted_position"]
    if "Position" in race_df.columns:
        cols.append("Position")

    display = race_df.sort_values("predicted_position")[cols].head(top_n)

    if "Position" in display.columns:
        display = display.rename(columns={"Position": "actual_position"})
        display["error"] = (
            display["predicted_position"] - display["actual_position"]
        ).abs()
        race_mae = display["error"].mean()
        header   = f"\n{'─'*55}\n  {year} {event_name}  (MAE = {race_mae:.2f})\n{'─'*55}"
    else:
        header = f"\n{'─'*55}\n  {year} {event_name}\n{'─'*55}"

    print(header)
    print(display.to_string(index=False))


def main() -> None:
    args = parse_args()

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode = True

    # ── Load model ────────────────────────────────────────────────────────
    model_path = Path(args.model) if args.model else MODELS_DIR / "ranker.pkl"
    if not model_path.exists():
        log.error("Model file not found: %s  – run scripts/train.py first.", model_path)
        sys.exit(1)

    log.info("Loading model from %s …", model_path)
    ranker: RaceRanker = RaceRanker.load(model_path)
    log.info("Loaded: %r", ranker)

    # ── Load dataset (uses existing caches) ──────────────────────────────
    log.info("Loading dataset for year %d …", args.year)
    _, pred_data = build_training_dataset(test_year=args.year)

    if args.event:
        pred_data = pred_data[pred_data["EventName"] == args.event]
        if pred_data.empty:
            log.error("No data found for event '%s' in %d.", args.event, args.year)
            sys.exit(1)

    # ── Predict ──────────────────────────────────────────────────────────
    drop_cols  = get_drop_columns(pred_data)
    X_pred     = pred_data.drop(columns=[c for c in drop_cols if c in pred_data.columns])

    # Align feature columns to what the model expects
    if ranker.feature_columns:
        X_pred = X_pred.reindex(columns=ranker.feature_columns)

    pred_data = pred_data.copy()
    pred_data["predicted_position"] = ranker.predict_positions(
        X_pred, pred_data["EventName"]
    )

    # ── Display ──────────────────────────────────────────────────────────
    for event_name, race_df in pred_data.groupby("EventName", sort=False):
        _display_race(race_df, event_name, args.year, args.top_n)

    # ── Overall MAE (if actuals available) ───────────────────────────────
    if "Position" in pred_data.columns:
        overall_mae = (
            pred_data["predicted_position"] - pred_data["Position"]
        ).abs().mean()
        print(f"\nOverall MAE ({args.year}): {overall_mae:.3f} positions")


if __name__ == "__main__":
    main()
