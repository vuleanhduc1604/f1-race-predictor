"""
Feature engineering for the F1 Race Predictor.

Classes
-------
FeatureExtractor   – Builds historical driver / team / circuit features with
                     strict past-only data (no leakage).
FeatureEngineer    – Adds rolling windows and championship context features.

Standalone helpers
------------------
add_qualifying_features(df, quali_df)  – Merge qualifying gap features.
add_relative_features(df)              – Add within-race normalised features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MIN_CIRCUIT_RACES, ROLLING_WINDOWS
from src.utils.helpers import is_dnf


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Compute per-row historical features from race results.

    All features are computed from races *before* the current event date to
    prevent any form of data leakage.

    Parameters
    ----------
    race_results : pd.DataFrame
        The full race-results dataframe (sorted by EventDate inside __init__).
    circuit_history : pd.DataFrame | None
        Additional historical circuit data to extend circuit stats back
        further in time.  Typically the same as race_results.
    """

    def __init__(
        self,
        race_results: pd.DataFrame,
        circuit_history: pd.DataFrame | None = None,
    ) -> None:
        self.race_results = race_results.sort_values("EventDate").copy()

        if circuit_history is not None and len(circuit_history):
            self.circuit_data = (
                pd.concat([circuit_history, race_results], ignore_index=True)
                .sort_values("EventDate")
            )
        else:
            self.circuit_data = self.race_results

    # ------------------------------------------------------------------
    # Driver features
    # ------------------------------------------------------------------

    def extract_driver_features(self) -> pd.DataFrame:
        """Career statistics for each driver up to (but not including) each race."""
        rows = []
        for _, race in self.race_results.iterrows():
            past = self.race_results[
                self.race_results["EventDate"] < race["EventDate"]
            ]
            driver_past = past[past["DriverId"] == race["DriverId"]]

            if len(driver_past):
                n = len(driver_past)
                feats = {
                    "driver_total_races":            n,
                    "driver_total_points_finishes":  (driver_past["Points"] > 0).sum(),
                    "driver_avg_finish_position":    driver_past["Position"].mean(),
                    "driver_avg_grid_position":      driver_past["GridPosition"].mean(),
                    "driver_avg_points_per_race":    driver_past["Points"].mean(),
                    "driver_dnf_rate":               driver_past["Status"].apply(is_dnf).sum() / n,
                    "driver_avg_positions_gained":   (driver_past["GridPosition"] - driver_past["Position"]).mean(),
                    "driver_overtake_success_rate":  (driver_past["Position"] < driver_past["GridPosition"]).sum() / n,
                    "driver_finish_position_stddev": driver_past["Position"].std(),
                }
            else:
                feats = {
                    "driver_total_races":            0,
                    "driver_total_points_finishes":  0,
                    "driver_avg_finish_position":    None,
                    "driver_avg_grid_position":      None,
                    "driver_avg_points_per_race":    0.0,
                    "driver_dnf_rate":               None,
                    "driver_avg_positions_gained":   None,
                    "driver_overtake_success_rate":  None,
                    "driver_finish_position_stddev": None,
                }

            row = race.to_dict()
            row.update(feats)
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"✓ Driver features extracted  ({len(df)} rows)")
        return df

    # ------------------------------------------------------------------
    # Team features
    # ------------------------------------------------------------------

    def extract_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical team performance up to each race."""
        rows = []
        for _, race in df.iterrows():
            past      = df[df["EventDate"] < race["EventDate"]]
            team_past = past[past["TeamId"] == race["TeamId"]]

            if len(team_past):
                n = len(team_past)
                feats = {
                    "team_avg_finish_position":  team_past["Position"].mean(),
                    "team_avg_points_per_race":  team_past["Points"].mean(),
                    "team_total_podiums":        (team_past["Position"] <= 3).sum(),
                    "team_dnf_rate":             team_past["Status"].apply(is_dnf).sum() / n,
                    "team_recent_avg_position":  (
                        team_past.tail(10)["Position"].mean()
                        if n >= 10 else team_past["Position"].mean()
                    ),
                    "team_recent_total_points":  (
                        team_past.tail(10)["Points"].sum()
                        if n >= 10 else team_past["Points"].sum()
                    ),
                }
            else:
                feats = {
                    "team_avg_finish_position": None,
                    "team_avg_points_per_race": 0.0,
                    "team_total_podiums":       0,
                    "team_dnf_rate":            None,
                    "team_recent_avg_position": None,
                    "team_recent_total_points": 0,
                }

            row = race.to_dict()
            row.update(feats)
            rows.append(row)

        df_out = pd.DataFrame(rows)
        print(f"✓ Team features extracted  ({len(df_out)} rows)")
        return df_out

    # ------------------------------------------------------------------
    # Driver–team synergy features
    # ------------------------------------------------------------------

    def extract_driver_team_synergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performance of a specific driver–team combination."""
        rows = []
        for _, race in df.iterrows():
            past = df[
                (df["EventDate"] < race["EventDate"])
                & (df["DriverId"] == race["DriverId"])
                & (df["TeamId"]   == race["TeamId"])
            ]

            if len(past):
                n = len(past)
                feats = {
                    "driver_team_avg_finish_position":  past["Position"].mean(),
                    "driver_team_avg_points_per_race":  past["Points"].mean(),
                    "driver_team_dnf_rate":             past["Status"].apply(is_dnf).sum() / n,
                    "driver_team_avg_positions_gained": (past["GridPosition"] - past["Position"]).mean(),
                    "driver_team_overtake_success_rate":(past["Position"] < past["GridPosition"]).sum() / n,
                }
            else:
                feats = {
                    "driver_team_avg_finish_position":  None,
                    "driver_team_avg_points_per_race":  0.0,
                    "driver_team_dnf_rate":             None,
                    "driver_team_avg_positions_gained": None,
                    "driver_team_overtake_success_rate":None,
                }

            row = race.to_dict()
            row.update(feats)
            rows.append(row)

        df_out = pd.DataFrame(rows)
        print(f"✓ Driver–team synergy features extracted  ({len(df_out)} rows)")
        return df_out

    # ------------------------------------------------------------------
    # Circuit features
    # ------------------------------------------------------------------

    def extract_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Historical circuit characteristics using self.circuit_data."""
        rows = []
        for _, race in df.iterrows():
            past = self.circuit_data[
                (self.circuit_data["EventDate"] < race["EventDate"])
                & (self.circuit_data["Location"] == race["Location"])
            ]

            if len(past) >= MIN_CIRCUIT_RACES:
                pos_changes = (past["GridPosition"] - past["Position"]).abs()
                pole        = past[past["GridPosition"] == 1]
                top3_grid   = past[past["GridPosition"] <= 3]

                feats = {
                    "circuit_avg_position_changes":    pos_changes.mean(),
                    "circuit_pole_win_rate":           (pole["Position"] == 1).mean() if len(pole) else None,
                    "circuit_top3_grid_podium_rate":   (top3_grid["Position"] <= 3).mean() if len(top3_grid) else None,
                    "circuit_grid_position_correlation":past["GridPosition"].corr(past["Position"]),
                    "circuit_avg_dnf_rate":            past["Status"].apply(is_dnf).mean(),
                    "circuit_races_in_history":        past["EventDate"].nunique(),
                }
            else:
                feats = {
                    "circuit_avg_position_changes":    None,
                    "circuit_pole_win_rate":           None,
                    "circuit_top3_grid_podium_rate":   None,
                    "circuit_grid_position_correlation":None,
                    "circuit_avg_dnf_rate":            None,
                    "circuit_races_in_history":        len(past),
                }

            row = race.to_dict()
            row.update(feats)
            rows.append(row)

        df_out = pd.DataFrame(rows)
        print(f"✓ Circuit features extracted  ({len(df_out)} rows)")
        return df_out

    # ------------------------------------------------------------------
    # Driver–circuit specialist features
    # ------------------------------------------------------------------

    def extract_driver_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """How well a driver performs at a specific circuit historically."""
        rows = []
        for _, race in df.iterrows():
            past = df[
                (df["EventDate"] < race["EventDate"])
                & (df["DriverId"] == race["DriverId"])
                & (df["Location"] == race["Location"])
            ]

            if len(past):
                n = len(past)
                feats = {
                    "driver_circuit_avg_finish_position":   past["Position"].mean(),
                    "driver_circuit_podiums":               (past["Position"] <= 3).sum(),
                    "driver_circuit_best_position":         past["Position"].min(),
                    "driver_circuit_avg_grid_position":     past["GridPosition"].mean(),
                    "driver_circuit_avg_positions_gained":  (past["GridPosition"] - past["Position"]).mean(),
                    "driver_circuit_overtake_success_rate": (past["Position"] < past["GridPosition"]).sum() / n,
                    "driver_circuit_dnf_rate":              past["Status"].apply(is_dnf).sum() / n,
                }
            else:
                feats = {
                    "driver_circuit_avg_finish_position":   None,
                    "driver_circuit_podiums":               0,
                    "driver_circuit_best_position":         None,
                    "driver_circuit_avg_grid_position":     None,
                    "driver_circuit_avg_positions_gained":  None,
                    "driver_circuit_overtake_success_rate": None,
                    "driver_circuit_dnf_rate":              None,
                }

            row = race.to_dict()
            row.update(feats)
            rows.append(row)

        df_out = pd.DataFrame(rows)
        print(f"✓ Driver–circuit features extracted  ({len(df_out)} rows)")
        return df_out

    # ------------------------------------------------------------------
    # Convenience: run full pipeline
    # ------------------------------------------------------------------

    def extract_all(self) -> pd.DataFrame:
        """
        Run all extraction steps in order and return the combined DataFrame.
        Equivalent to calling extract_driver → extract_team → ... in sequence.
        """
        df = self.extract_driver_features()
        df = self.extract_team_features(df)
        df = self.extract_driver_team_synergy(df)
        df = self.extract_circuit_features(df)
        df = self.extract_driver_circuit_features(df)
        return df


# ---------------------------------------------------------------------------
# FeatureEngineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Add time-series features that summarise recent form and championship context.

    Parameters
    ----------
    windows : list[int]
        Rolling window sizes (number of races) to use for form features.
    """

    def __init__(self, windows: list[int] | None = None) -> None:
        self.windows = windows or ROLLING_WINDOWS

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-driver rolling averages for position, points, DNF, podiums, positions gained."""
        df = df.copy().sort_values(["DriverId", "EventDate"])
        df["_is_dnf"] = df["Status"].apply(is_dnf)

        for w in self.windows:
            grp = df.groupby("DriverId")
            df[f"driver_avg_positions_last_{w}"] = grp["Position"].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )
            df[f"driver_total_points_last_{w}"] = grp["Points"].transform(
                lambda x: x.rolling(w, min_periods=1).sum().shift(1)
            )
            df[f"driver_avg_dnf_rate_last_{w}"] = grp["_is_dnf"].transform(
                lambda x: x.rolling(w, min_periods=1).mean().shift(1)
            )
            df[f"driver_podium_count_last_{w}"] = grp["Position"].transform(
                lambda x: (x <= 3).rolling(w, min_periods=1).sum().shift(1)
            )
            s = df["GridPosition"] - df["Position"]
            df[f"driver_avg_position_gained_last_{w}"] = (
                s.groupby(df["DriverId"])
                .transform(lambda x: x.rolling(w, min_periods=1).mean().shift(1))
            )

        df.drop(columns=["_is_dnf"], inplace=True)
        return df

    def add_championship_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cumulative points / championship position features, all shifted one race
        to prevent leakage.
        """
        df = df.copy().sort_values("EventDate")

        # Cumulative points before each race
        df["points_before_race"] = (
            df.groupby("DriverId")["Points"]
            .transform(lambda x: x.cumsum().shift(1).fillna(0))
        )

        # Championship position, gap to leader, gap to next
        gbe = df.groupby("EventDate")
        df["championship_position_before_race"] = gbe["points_before_race"].transform(
            lambda x: x.rank(ascending=False, method="min")
        )
        df["points_gap_to_leader_before_race"] = gbe["points_before_race"].transform(
            lambda x: x.max() - x
        )
        df["points_gap_to_next_before_race"] = gbe["points_before_race"].transform(
            lambda x: x.sort_values(ascending=False).diff(-1).fillna(0)
        )

        # Drought features (races since last win / podium / points)
        df["races_since_last_win"] = (
            df.groupby("DriverId")["Position"]
            .transform(lambda x: (x != 1).cumsum()
                       - (x != 1).cumsum().where(x == 1).ffill().fillna(0))
        )
        df["races_since_last_podium"] = (
            df.groupby("DriverId")["Position"]
            .transform(lambda x: (x > 3).cumsum()
                       - (x > 3).cumsum().where(x <= 3).ffill().fillna(0))
        )
        df["races_since_last_points_finish"] = (
            df.groupby("DriverId")["Points"]
            .transform(lambda x: (x == 0).cumsum()
                       - (x == 0).cumsum().where(x > 0).ffill().fillna(0))
        )

        # Race number in season (sequential per driver; used as proxy for tyre/car evolution)
        df["race_number_in_season"] = (
            df.groupby(["Year", "EventDate"]).cumcount() + 1
        )

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply both rolling and championship steps in one call."""
        df = self.add_rolling_features(df)
        df = self.add_championship_context(df)
        return df


# ---------------------------------------------------------------------------
# Qualifying gap features
# ---------------------------------------------------------------------------

def add_qualifying_features(df: pd.DataFrame, quali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge qualifying gap features into *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Race dataset with columns: Abbreviation, Year, EventName.
    quali_df : pd.DataFrame
        Output of the qualifying cache with columns:
        Abbreviation, Year, EventName, Q1_s, Q2_s, Q3_s.

    Returns
    -------
    pd.DataFrame with four new columns:
        q3_gap_to_pole  – Q3 time minus pole time (NaN if not in Q3)
        q_best_gap      – Driver's best qualifying time minus pole time
        reached_q3      – Binary flag: driver participated in Q3
        reached_q2      – Binary flag: driver participated in Q2
    """
    quali = quali_df.copy()

    def _best_time(row):
        if pd.notna(row["Q3_s"]): return row["Q3_s"]
        if pd.notna(row["Q2_s"]): return row["Q2_s"]
        return row["Q1_s"]

    quali["q_best_time"] = quali.apply(_best_time, axis=1)

    pole = (
        quali.groupby(["Year", "EventName"])["Q3_s"]
        .min()
        .rename("q3_pole_time")
        .reset_index()
    )
    quali = quali.merge(pole, on=["Year", "EventName"])

    quali["q3_gap_to_pole"] = np.where(
        quali["Q3_s"].notna(),
        quali["Q3_s"] - quali["q3_pole_time"],
        np.nan,
    )
    quali["q_best_gap"] = quali["q_best_time"] - quali["q3_pole_time"]
    quali["reached_q3"] = quali["Q3_s"].notna().astype(int)
    quali["reached_q2"] = quali["Q2_s"].notna().astype(int)

    quali_features = quali[
        ["Abbreviation", "Year", "EventName",
         "q3_gap_to_pole", "q_best_gap", "reached_q3", "reached_q2"]
    ]

    return df.merge(quali_features, on=["Abbreviation", "Year", "EventName"], how="left")


# ---------------------------------------------------------------------------
# Within-race relative features
# ---------------------------------------------------------------------------

_RELATIVE_FEATURE_MAP = [
    ("driver_avg_positions_last_5",           "relative_driver_form_5"),
    ("driver_avg_positions_last_10",          "relative_driver_form_10"),
    ("team_recent_avg_position",              "relative_team_form"),
    ("driver_total_points_last_10",           "relative_driver_points_10"),
    ("driver_total_points_last_5",            "relative_driver_points_5"),
    ("championship_position_before_race",     "relative_champ_pos"),
]


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that express each driver's metrics relative to the field
    median in the same race.

    This normalisation gives the model within-race context without introducing
    any data leakage (the underlying features are already computed from past
    data only).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with Year, EventName, and the source feature columns.

    Returns
    -------
    pd.DataFrame with new relative_* columns appended.
    """
    df = df.copy()
    race_key = ["Year", "EventName"]
    for src_col, new_col in _RELATIVE_FEATURE_MAP:
        if src_col in df.columns:
            median_in_race = df.groupby(race_key)[src_col].transform("median")
            df[new_col]    = df[src_col] - median_in_race
    return df
