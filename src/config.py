"""
Central configuration for the F1 Race Predictor project.
All constants, paths, and default hyperparameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# FastF1 cache sub-files
RAW_RESULTS_CACHE  = CACHE_DIR / "raw_results_2018_2025.pkl"
PRACTICE_CACHE     = CACHE_DIR / "practice_features.pkl"
QUALIFYING_CACHE   = CACHE_DIR / "qualifying_results.pkl"
FEATURES_CACHE     = CACHE_DIR / "data_with_all_features.pkl"

# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
TRAINING_YEARS = list(range(2018, 2026))   # 2018–2025 inclusive
TEST_YEAR      = 2025

# ---------------------------------------------------------------------------
# Practice session extraction
# ---------------------------------------------------------------------------
COMPOUND_MAP: dict[str, str] = {
    "HYPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",
    "SUPERSOFT": "SOFT",
    "SOFT":      "SOFT",
    "MEDIUM":    "MEDIUM",
    "HARD":      "HARD",
}

MIN_DRIVERS         = 15
MIN_LAPS_PER_DRIVER = 3

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
ROLLING_WINDOWS   = [5, 10]
MIN_CIRCUIT_RACES = 3       # Minimum historical races to compute circuit features

# ---------------------------------------------------------------------------
# Model mode: "regressor" (default) or "ranker"
# ---------------------------------------------------------------------------
DEFAULT_MODE = "regressor"

# ---------------------------------------------------------------------------
# Default LGBMRegressor hyperparameters (from notebook Optuna tuning)
# ---------------------------------------------------------------------------
DEFAULT_REGRESSOR_PARAMS: dict = {
    "n_estimators":      461,
    "max_depth":         4,
    "num_leaves":        21,
    "learning_rate":     0.010649053762318842,
    "min_child_samples": 13,
    "subsample":         0.9485808150015104,
    "colsample_bytree":  0.8694628677580424,
    "reg_alpha":         0.705267156184453,
    "reg_lambda":        0.0024099824381398805,
    "random_state":      42,
    "verbose":           -1,
}

# ---------------------------------------------------------------------------
# Default LGBMRanker hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_RANKER_PARAMS: dict = {
    "objective":         "lambdarank",
    "n_estimators":      600,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        40,
    "min_child_samples": 15,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.01,
    "reg_lambda":        0.05,
    "random_state":      42,
    "verbose":           -1,
}

# ---------------------------------------------------------------------------
# Feature selection: columns to drop before model training
# ---------------------------------------------------------------------------

# Metadata / target / leaky columns  (matches notebook drop_columns_v1)
DROP_METADATA = [
    "DriverNumber", "BroadcastName", "Abbreviation", "FirstName", "LastName",
    "FullName", "HeadshotUrl", "CountryCode", "TeamColor", "EventName",
    "OfficialEventName", "EventFormat", "Country", "Position", "RoundNumber",
    "ClassifiedPosition", "Points", "Time", "Status", "Laps",
    "Q1", "Q2", "Q3",
    "Location", "EventDate", "Year", "TeamName", "DriverId", "TeamId",
    "RaceDateTime",
]

# Low-importance and high-missing-rate features (matches notebook drop_low_importance)
DROP_LOW_IMPORTANCE = [
    # Zero importance
    "fp1_max_speed_fl", "fp1_max_speed_i2", "fp1_best_lap_delta",
    "fp1_pb_lap_delta", "fp1_best_lap_pct_off", "driver_circuit_podiums",
    "has_sprint",
    # Near-zero importance
    "fp1_s1_delta", "fp1_s2_delta", "fp1_s3_delta",
    "fp1_best_soft_delta", "fp1_best_medium_delta", "fp1_best_hard_delta",
    "fp1_max_speed_i1", "fp1_max_speed_st",
    "driver_circuit_best_position",
    "fp2_pb_lap_delta", "fp2_max_speed_fl",
    "fp3_max_speed_i1", "fp3_max_speed_fl",
]

# Historical form features: rolling-window stats and drought counters.
# These are less reliable after major regulation changes (e.g. 2026) because
# past performance may not reflect current car/driver competitiveness.
#
# To re-enable: remove DROP_HISTORICAL_FORM from get_drop_columns() in helpers.py.
DROP_HISTORICAL_FORM = [
    # Rolling averages / sums (last 5 and last 10 races)
    "driver_avg_positions_last_5",       "driver_avg_positions_last_10",
    "driver_total_points_last_5",        "driver_total_points_last_10",
    "driver_avg_dnf_rate_last_5",        "driver_avg_dnf_rate_last_10",
    "driver_podium_count_last_5",        "driver_podium_count_last_10",
    "driver_avg_position_gained_last_5", "driver_avg_position_gained_last_10",
    # Drought counters
    "races_since_last_win",
    "races_since_last_podium",
    "races_since_last_points_finish",
]

# Columns added to test_data / training_data at prediction time (not features)
DROP_PREDICTIONS = [
    "predicted_score", "predicted_rank",
    "ranker_score", "ranker_position",
    "tuned_ranker_score", "tuned_ranker_position",
    "winner_probability", "predicted_winner",
    "podium_probability", "predicted_podium",
    "is_winner", "is_podium",
]
