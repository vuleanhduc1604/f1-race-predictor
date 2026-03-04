"""
RaceRanker – wrapper around LightGBM that supports two modes:

  mode="regressor"  (default)
      Uses LGBMRegressor to predict a continuous position value.
      Within each race, drivers are ranked ascending (lower value → P1).
      No group information is required.

  mode="ranker"
      Uses LGBMRanker (LambdaMART) to learn a relevance ordering.
      Within each race, drivers are ranked descending (higher score → P1).
      Requires per-race group sizes passed to fit().

Switch via the --mode flag in scripts/train.py, or by passing
mode="ranker" to RaceRanker() directly to revert to ranking behaviour.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import DEFAULT_MODE, DEFAULT_RANKER_PARAMS, DEFAULT_REGRESSOR_PARAMS
from src.utils.helpers import get_drop_columns, get_race_groups, positions_to_relevance


class RaceRanker:
    """
    Predict F1 race finishing positions using LightGBM.

    Parameters
    ----------
    params : dict | None
        LightGBM hyperparameters.  Defaults to DEFAULT_REGRESSOR_PARAMS when
        mode="regressor" or DEFAULT_RANKER_PARAMS when mode="ranker".
    mode : str
        "regressor" (default) or "ranker".

    Examples
    --------
    >>> ranker = RaceRanker()                        # LGBMRegressor mode
    >>> ranker.fit(X_train, y_pos_train)             # no groups needed
    >>> mae = ranker.evaluate(X_test, y_pos_test, test_data["EventName"])

    >>> ranker = RaceRanker(mode="ranker")           # LambdaMART mode
    >>> ranker.fit(X_train, y_pos_train, train_groups)
    """

    def __init__(
        self,
        params: dict | None = None,
        mode: str = DEFAULT_MODE,
    ) -> None:
        if mode not in ("regressor", "ranker"):
            raise ValueError(f"mode must be 'regressor' or 'ranker', got '{mode}'")
        self.mode = mode
        if params is None:
            default = DEFAULT_REGRESSOR_PARAMS if mode == "regressor" else DEFAULT_RANKER_PARAMS
            self.params = default.copy()
        else:
            self.params = params.copy()
        self.model: lgb.LGBMRegressor | lgb.LGBMRanker | None = None
        self.feature_columns: list[str] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_position: pd.Series,
        groups: list[int] | None = None,
    ) -> "RaceRanker":
        """
        Train the model.

        Parameters
        ----------
        X          : Feature matrix (rows sorted by EventDate/EventName).
        y_position : Integer race finishing positions (1 = first).
        groups     : Per-race driver counts (required for mode="ranker").
        """
        if self.mode == "regressor":
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y_position.astype(float))
        else:
            if groups is None:
                raise ValueError("groups must be provided when mode='ranker'")
            y_relevance = positions_to_relevance(y_position)
            self.model  = lgb.LGBMRanker(**self.params)
            self.model.fit(X, y_relevance, group=groups)

        self.feature_columns = list(X.columns)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw model scores."""
        if self.model is None:
            raise ValueError("Model has not been trained.  Call fit() first.")
        return self.model.predict(X)

    def predict_positions(
        self,
        X: pd.DataFrame,
        event_names: pd.Series,
    ) -> np.ndarray:
        """
        Convert raw scores to integer 1-based positions within each race.

        Regressor mode  : lower predicted value → better position (rank ascending).
        Ranker mode     : higher score → better position (rank descending).

        Parameters
        ----------
        X           : Feature matrix.
        event_names : pd.Series aligned with X giving each row's EventName.

        Returns
        -------
        np.ndarray of integer positions.
        """
        scores     = self.predict_scores(X)
        ascending  = (self.mode == "regressor")
        tmp        = pd.DataFrame({"EventName": event_names.values, "score": scores})
        tmp["position"] = (
            tmp.groupby("EventName")["score"]
            .rank(method="first", ascending=ascending)
            .astype(int)
        )
        return tmp["position"].values

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: pd.DataFrame,
        y_position: pd.Series,
        event_names: pd.Series,
    ) -> float:
        """Return Mean Absolute Error of predicted vs actual positions."""
        preds = self.predict_positions(X, event_names)
        return float(mean_absolute_error(y_position.values, preds))

    def evaluation_report(
        self,
        df: pd.DataFrame,
        drop_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Per-race MAE report.

        Parameters
        ----------
        df        : Full test dataframe (must have Position, EventName, Year
                    and all feature columns).
        drop_cols : Columns to exclude when building the feature matrix.
                    If None, uses get_drop_columns(df).

        Returns
        -------
        pd.DataFrame with columns: Year, EventName, drivers, MAE, sorted by
        MAE descending.
        """
        drop_cols = drop_cols or get_drop_columns(df)
        X         = df.drop(columns=[c for c in drop_cols if c in df.columns])

        preds = self.predict_positions(X, df["EventName"])
        df    = df.copy()
        df["_pred"] = preds
        df["_err"]  = (df["_pred"] - df["Position"]).abs()

        report = (
            df.groupby(["Year", "EventName"])
            .agg(drivers=("Position", "count"), MAE=("_err", "mean"))
            .reset_index()
            .sort_values("MAE", ascending=False)
        )
        return report

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> pd.DataFrame:
        """Return a DataFrame of feature importances sorted descending."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        return (
            pd.DataFrame({
                "Feature":    self.feature_columns,
                "Importance": self.model.feature_importances_,
            })
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the full RaceRanker object (model + metadata) to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "RaceRanker":
        """Load a previously saved RaceRanker from *path*."""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def __repr__(self) -> str:
        trained = self.model is not None
        n_feat  = len(self.feature_columns) if self.feature_columns else 0
        return (
            f"RaceRanker(mode={self.mode!r}, trained={trained}, "
            f"n_features={n_feat}, "
            f"n_estimators={self.params.get('n_estimators')})"
        )
