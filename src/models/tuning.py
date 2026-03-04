"""
Hyperparameter optimisation for RaceRanker using Optuna.

Key design decisions
--------------------
* Season-based cross-validation (leave-one-year-out) is used instead of
  random k-fold to respect the temporal structure of the data.
* The CV objective is MAE of within-race predicted positions vs actual
  positions — the same metric used for final evaluation.
* mode="regressor" (default) trains LGBMRegressor without groups.
* mode="ranker" trains LGBMRanker with lambdarank + group sizes.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.utils.helpers import get_drop_columns, positions_to_relevance

log = logging.getLogger(__name__)

# Seasons used as validation folds in leave-one-year-out CV
DEFAULT_CV_YEARS = [2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def regressor_cv_mae(
    params: dict,
    train_data: pd.DataFrame,
    drop_cols: list[str],
    cv_years: list[int] | None = None,
) -> float:
    """
    Evaluate *params* for LGBMRegressor using leave-one-year-out CV.

    Parameters
    ----------
    params     : LGBMRegressor hyperparameters.
    train_data : Full training DataFrame (sorted by EventDate/EventName).
    drop_cols  : Columns to remove before training/validation.
    cv_years   : Years to use as validation folds.

    Returns
    -------
    float – Mean MAE across all CV folds.
    """
    cv_years  = cv_years or DEFAULT_CV_YEARS
    fold_maes: list[float] = []

    for val_year in cv_years:
        tr = train_data[train_data["Year"] != val_year].copy()
        va = train_data[train_data["Year"] == val_year].copy()

        if tr.empty or va.empty:
            continue

        X_tr = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
        y_tr = tr["Position"].astype(float)

        X_va = va.drop(columns=[c for c in drop_cols if c in va.columns])
        X_va = X_va.reindex(columns=X_tr.columns)

        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
        model.fit(X_tr, y_tr)

        scores = model.predict(X_va)
        va     = va.copy()
        va["_score"]    = scores
        va["_pred_pos"] = (
            va.groupby("EventName")["_score"]
            .rank(method="first", ascending=True)
            .astype(int)
        )
        fold_maes.append(
            mean_absolute_error(va["Position"].values, va["_pred_pos"].values)
        )

    return float(np.mean(fold_maes)) if fold_maes else float("inf")


def ranker_cv_mae(
    params: dict,
    train_data: pd.DataFrame,
    drop_cols: list[str],
    cv_years: list[int] | None = None,
) -> float:
    """
    Evaluate *params* for LGBMRanker using leave-one-year-out CV.

    Parameters
    ----------
    params     : LGBMRanker hyperparameters (must include 'objective').
    train_data : Full training DataFrame (sorted by EventDate/EventName).
    drop_cols  : Columns to remove before training/validation.
    cv_years   : Years to use as validation folds.

    Returns
    -------
    float – Mean MAE across all CV folds.
    """
    cv_years   = cv_years or DEFAULT_CV_YEARS
    fold_maes: list[float] = []

    for val_year in cv_years:
        tr = train_data[train_data["Year"] != val_year].copy()
        va = train_data[train_data["Year"] == val_year].copy()

        if tr.empty or va.empty:
            continue

        tr_groups = tr.groupby(["Year", "EventName"], sort=False).size().values
        X_tr       = tr.drop(columns=[c for c in drop_cols if c in tr.columns])
        y_tr_rel   = positions_to_relevance(tr["Position"].astype(float))

        X_va       = va.drop(columns=[c for c in drop_cols if c in va.columns])
        X_va       = X_va.reindex(columns=X_tr.columns)

        model = lgb.LGBMRanker(**params, verbose=-1, random_state=42)
        model.fit(X_tr, y_tr_rel, group=tr_groups)

        scores = model.predict(X_va)
        va     = va.copy()
        va["_score"]    = scores
        va["_pred_pos"] = (
            va.groupby("EventName")["_score"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        fold_maes.append(
            mean_absolute_error(va["Position"].values, va["_pred_pos"].values)
        )

    return float(np.mean(fold_maes)) if fold_maes else float("inf")


# ---------------------------------------------------------------------------
# Optuna objectives
# ---------------------------------------------------------------------------

def _make_regressor_objective(
    train_data: pd.DataFrame,
    drop_cols: list[str],
    cv_years: list[int] | None,
):
    """Factory that returns an Optuna objective for LGBMRegressor."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      200, 1000),
            "max_depth":         trial.suggest_int("max_depth",          3, 8),
            "num_leaves":        trial.suggest_int("num_leaves",        15, 80),
            "learning_rate":     trial.suggest_float("learning_rate",   0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples",  5, 40),
            "subsample":         trial.suggest_float("subsample",        0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-4, 1.0, log=True),
        }
        return regressor_cv_mae(params, train_data, drop_cols, cv_years)

    return objective


def _make_ranker_objective(
    train_data: pd.DataFrame,
    drop_cols: list[str],
    cv_years: list[int] | None,
):
    """Factory that returns an Optuna objective for LGBMRanker."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":         "lambdarank",
            "n_estimators":      trial.suggest_int("n_estimators",      200, 1000),
            "max_depth":         trial.suggest_int("max_depth",          3, 8),
            "num_leaves":        trial.suggest_int("num_leaves",        15, 80),
            "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples",  5,  40),
            "subsample":         trial.suggest_float("subsample",        0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",        1e-4, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",       1e-4, 1.0, log=True),
        }
        return ranker_cv_mae(params, train_data, drop_cols, cv_years)

    return objective


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hyperparameter_search(
    train_data: pd.DataFrame,
    drop_cols: list[str] | None = None,
    n_trials: int = 50,
    cv_years: list[int] | None = None,
    show_progress: bool = True,
    mode: str = "regressor",
) -> dict:
    """
    Run Optuna hyperparameter search and return the best parameter dict.

    Parameters
    ----------
    train_data    : Full training DataFrame (output of build_training_dataset).
    drop_cols     : Columns to exclude when building feature matrices.
                    Defaults to get_drop_columns(train_data).
    n_trials      : Number of Optuna trials.
    cv_years      : Validation years for leave-one-year-out CV.
    show_progress : Show tqdm progress bar.
    mode          : "regressor" or "ranker".

    Returns
    -------
    dict  – Best hyperparameters (includes 'objective': 'lambdarank' when
            mode="ranker").
    """
    if mode not in ("regressor", "ranker"):
        raise ValueError(f"mode must be 'regressor' or 'ranker', got '{mode}'")

    drop_cols = drop_cols or get_drop_columns(train_data)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if mode == "regressor":
        objective_fn = _make_regressor_objective(train_data, drop_cols, cv_years)
    else:
        objective_fn = _make_ranker_objective(train_data, drop_cols, cv_years)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    best = dict(study.best_params)
    if mode == "ranker":
        best["objective"] = "lambdarank"

    log.info("Best CV MAE (%s): %.4f", mode, study.best_value)
    log.info("Best params: %s", best)
    return best
