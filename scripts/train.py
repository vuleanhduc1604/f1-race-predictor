#!/usr/bin/env python
"""
Train the F1 race-position model.

Usage
-----
    python scripts/train.py                         # LGBMRegressor, 50 Optuna trials
    python scripts/train.py --mode ranker           # switch to LGBMRanker (LambdaMART)
    python scripts/train.py --n-trials 100          # more tuning
    python scripts/train.py --skip-tuning           # use default params directly
    python scripts/train.py --force-rebuild         # ignore all caches
    python scripts/train.py --all-years             # train one leak-free model per year (2019–TEST_YEAR)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1

from src.config import (
    CACHE_DIR,
    DEFAULT_MODE,
    DEFAULT_RANKER_PARAMS,
    DEFAULT_REGRESSOR_PARAMS,
    MODELS_DIR,
    TEST_YEAR,
)
from src.data.loaders import build_training_dataset
from src.models.ranker import RaceRanker
from src.models.tuning import run_hyperparameter_search
from src.utils.helpers import get_drop_columns, get_race_groups

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the F1 race-position model.")
    p.add_argument("--mode",        type=str,  default=DEFAULT_MODE,
                   choices=["regressor", "ranker"],
                   help="Model type: 'regressor' (default) or 'ranker' (LambdaMART).")
    p.add_argument("--n-trials",    type=int,  default=50,
                   help="Number of Optuna HPO trials (default: 50).")
    p.add_argument("--skip-tuning", action="store_true",
                   help="Skip Optuna and use default params for the chosen mode.")
    p.add_argument("--force-rebuild", action="store_true",
                   help="Re-extract all features, ignoring existing caches.")
    p.add_argument("--test-year",   type=int,  default=TEST_YEAR,
                   help=f"Hold-out season for evaluation (default: {TEST_YEAR}).")
    p.add_argument("--model-name",  type=str,  default="ranker.pkl",
                   help="Filename to save the trained model under models/.")
    p.add_argument("--all-years",   action="store_true",
                   help=(
                       "Train one model per test year from 2019 to TEST_YEAR, saving each as "
                       "ranker_test{year}.pkl. Implies --skip-tuning. Use these models in the "
                       "API to eliminate in-sample data leakage for historical predictions."
                   ))
    return p.parse_args()


def _train_one(args: argparse.Namespace, test_year: int, model_name: str) -> None:
    """Train and save a single model for *test_year*."""
    log.info("━━━ Training model for test_year=%d → %s ━━━", test_year, model_name)

    train_data, test_data = build_training_dataset(
        test_year     = test_year,
        force_rebuild = args.force_rebuild,
    )

    drop_cols = get_drop_columns(train_data)
    X_train   = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    y_train   = train_data["Position"].astype(float)
    X_test    = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
    X_test    = X_test.reindex(columns=X_train.columns)
    y_test    = test_data["Position"].astype(float)

    log.info("Train: %d rows, %d features  |  Test: %d rows",
             len(X_train), X_train.shape[1], len(X_test))

    if args.skip_tuning or args.all_years:
        default_params = DEFAULT_REGRESSOR_PARAMS if args.mode == "regressor" else DEFAULT_RANKER_PARAMS
        best_params = default_params.copy()
    else:
        best_params = run_hyperparameter_search(
            train_data,
            drop_cols = drop_cols,
            n_trials  = args.n_trials,
            mode      = args.mode,
        )

    ranker = RaceRanker(params=best_params, mode=args.mode)
    if args.mode == "ranker":
        ranker.fit(X_train, y_train, get_race_groups(train_data))
    else:
        ranker.fit(X_train, y_train)

    if len(X_test):
        mae = ranker.evaluate(X_test, y_test, test_data["EventName"])
        log.info("Hold-out MAE (%d season): %.3f positions", test_year, mae)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ranker.save(MODELS_DIR / model_name)
    log.info("Model saved → %s", MODELS_DIR / model_name)


def main() -> None:
    args = parse_args()

    # FastF1 cache
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode = True

    log.info("Mode: %s", args.mode)

    if args.all_years:
        # Train one leak-free model per test year: ranker_test2019.pkl … ranker_test2025.pkl
        years = list(range(2019, TEST_YEAR + 1))
        log.info("--all-years: training %d models (%d → %d), skip-tuning=True",
                 len(years), years[0], years[-1])
        for year in years:
            _train_one(args, test_year=year, model_name=f"ranker_test{year}.pkl")
        log.info("All %d models trained successfully.", len(years))
        return

    # ── Single model (default) ────────────────────────────────────────────
    # ── 1. Build dataset ──────────────────────────────────────────────────
    log.info("Building training dataset …")
    train_data, test_data = build_training_dataset(
        test_year     = args.test_year,
        force_rebuild = args.force_rebuild,
    )

    drop_cols = get_drop_columns(train_data)
    X_train   = train_data.drop(columns=[c for c in drop_cols if c in train_data.columns])
    y_train   = train_data["Position"].astype(float)

    X_test    = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
    X_test    = X_test.reindex(columns=X_train.columns)
    y_test    = test_data["Position"].astype(float)

    log.info("Train: %d rows, %d features  |  Test: %d rows",
             len(X_train), X_train.shape[1], len(X_test))

    # ── 2. Hyperparameter tuning ──────────────────────────────────────────
    if args.skip_tuning:
        default_params = DEFAULT_REGRESSOR_PARAMS if args.mode == "regressor" else DEFAULT_RANKER_PARAMS
        log.info("Skipping Optuna; using default params for mode=%s.", args.mode)
        best_params = default_params.copy()
    else:
        log.info("Running Optuna HPO (%d trials, mode=%s) …", args.n_trials, args.mode)
        best_params = run_hyperparameter_search(
            train_data,
            drop_cols = drop_cols,
            n_trials  = args.n_trials,
            mode      = args.mode,
        )

    # ── 3. Train final model ──────────────────────────────────────────────
    log.info("Training final RaceRanker (mode=%s) …", args.mode)
    ranker = RaceRanker(params=best_params, mode=args.mode)

    if args.mode == "ranker":
        train_groups = get_race_groups(train_data)
        ranker.fit(X_train, y_train, train_groups)
    else:
        ranker.fit(X_train, y_train)

    # ── 4. Evaluate on hold-out season ────────────────────────────────────
    if len(X_test):
        mae = ranker.evaluate(X_test, y_test, test_data["EventName"])
        log.info("Hold-out MAE (%d season): %.3f positions", args.test_year, mae)

        report = ranker.evaluation_report(test_data, drop_cols=drop_cols)
        log.info("Per-race MAE (worst 5):\n%s",
                 report.head(5).to_string(index=False))
    else:
        log.info("No test data for %d yet — skipping evaluation.", args.test_year)

    # ── 5. Save model ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / args.model_name
    ranker.save(model_path)
    log.info("Model saved → %s", model_path)

    # ── 6. Feature importance summary ────────────────────────────────────
    top10 = ranker.feature_importances.head(10)
    log.info("Top-10 features:\n%s", top10.to_string(index=False))


if __name__ == "__main__":
    main()
