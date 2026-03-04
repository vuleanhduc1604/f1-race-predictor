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
    "n_estimators":      439,
    "max_depth":         4,
    "num_leaves":        29,
    "learning_rate":     0.011808313796154926,
    "min_child_samples": 24,
    "subsample":         0.7232403994453361,
    "colsample_bytree":  0.8063601537537767,
    "reg_alpha":         0.004348058550191909,
    "reg_lambda":        0.00024234756076934005,
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

# Metadata / target / leaky columns
DROP_METADATA = [
    "DriverNumber", "BroadcastName", "Abbreviation", "FirstName", "LastName",
    "FullName", "HeadshotUrl", "CountryCode", "TeamColor", "EventName",
    "OfficialEventName", "EventFormat", "Country", "Position", "RoundNumber",
    "ClassifiedPosition", "Points", "Time", "Status", "Laps",
    "Q1", "Q2", "Q3",               # raw timedelta qualy cols (always null in race results)
    "Location", "EventDate", "Year", "TeamName", "DriverId", "TeamId",
    "RaceDateTime",
]

# Near-zero importance features identified from notebook analysis
DROP_LOW_IMPORTANCE = [
    "fp1_best_lap_delta", "fp1_best_lap_pct_off", "fp1_pb_lap_delta",
    "fp1_s1_delta", "fp1_s2_delta", "fp1_max_speed_i2", "fp1_max_speed_fl",
    "fp1_best_hard_delta", "fp2_pb_lap_delta", "has_sprint",
    "driver_podium_count_last_5", "driver_circuit_podiums",
    "driver_circuit_best_position",
]

# Practice features with >80 % NaN rate (add noise rather than signal)
DROP_HIGH_MISSING = [
    "fp3_best_hard_delta",   # 96 % NaN
    "fp1_best_soft_delta",   # 95 % NaN
    "fp1_best_medium_delta", # 95 % NaN
    "fp1_max_speed_st",      # 93 % NaN
    "fp1_max_speed_i1",      # 93 % NaN
    "fp1_s3_delta",          # 93 % NaN
    "fp2_best_hard_delta",   # 84 % NaN
    "fp3_best_medium_delta", # 81 % NaN
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
