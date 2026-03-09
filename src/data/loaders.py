"""
Data loading and dataset construction.

Data flow
---------
1. load_raw_results()      – Race results 2018–2025 from FastF1 (cached).
2. load_practice_features()– FP1/FP2/FP3 pace features (cached).
3. load_qualifying_features()– Q1/Q2/Q3 times in seconds (cached).
4. build_training_dataset()– Orchestrates all of the above, applies feature
                             engineering, and returns train / test splits ready
                             for the model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from src.config import (
    CACHE_DIR,
    FEATURES_CACHE,
    PRACTICE_CACHE,
    QUALIFYING_CACHE,
    RAW_RESULTS_CACHE,
    ROLLING_WINDOWS,
    TEST_YEAR,
    TRAINING_YEARS,
)
from src.data.extractors import practiceExtractor, qualifyingExtractor
from src.data.features import (
    FeatureEngineer,
    FeatureExtractor,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Raw race results
# ---------------------------------------------------------------------------

def load_raw_results(
    years: list[int] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Load race results for *years* from FastF1, using a pickle cache.

    Parameters
    ----------
    years      : List of seasons to load.  Defaults to TRAINING_YEARS.
    cache_path : Override the default cache file path.
    force      : Re-fetch from FastF1 even if the cache exists.

    Returns
    -------
    pd.DataFrame with one row per driver-race result.
    """
    years      = years      or TRAINING_YEARS
    cache_path = cache_path or RAW_RESULTS_CACHE

    if cache_path.exists() and not force:
        log.info("Loading raw results from cache: %s", cache_path)
        return pd.read_pickle(cache_path)

    log.info("Fetching raw results from FastF1 for years %s …", years)
    all_races: list[pd.DataFrame] = []

    for year in years:
        log.info("  Season %d", year)
        schedule    = fastf1.get_event_schedule(year)
        race_events = schedule[schedule["EventFormat"] != "testing"]

        for _, event in race_events.iterrows():
            event_name = event["EventName"]
            round_num  = event["RoundNumber"]
            try:
                session = fastf1.get_session(year, event_name, "Race")
                session.load()
                results = session.results.copy()
                if results is None or len(results) == 0:
                    continue

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

                all_races.append(results)
                log.info("    ✓ Round %2d: %-30s – %d drivers",
                         round_num, event_name, len(results))

            except Exception as exc:
                log.warning("    ✗ Round %2d: %-30s – %s",
                            round_num, event_name, str(exc)[:60])

    if not all_races:
        raise ValueError("No race data loaded.  Check FastF1 cache / connection.")

    df = pd.concat(all_races, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_path)
    log.info("Saved %d results to %s", len(df), cache_path)
    return df


# ---------------------------------------------------------------------------
# Practice features
# ---------------------------------------------------------------------------

def load_practice_features(
    years: list[int] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract FP1 / FP2 / FP3 pace features for all events in *years*.

    Returns
    -------
    pd.DataFrame with columns: Driver, Year, EventName, <feature cols>
    """
    years      = years      or TRAINING_YEARS
    cache_path = cache_path or PRACTICE_CACHE

    if cache_path.exists() and not force:
        log.info("Loading practice features from cache: %s", cache_path)
        return pd.read_pickle(cache_path)

    log.info("Extracting practice features for years %s …", years)
    all_rows: list[pd.DataFrame] = []

    for year in years:
        log.info("  Season %d", year)
        schedule    = fastf1.get_event_schedule(year)
        race_events = schedule[schedule["EventFormat"] != "testing"]

        for _, event in race_events.iterrows():
            event_name   = event["EventName"]
            event_format = event["EventFormat"]
            is_sprint    = event_format in ("sprint", "sprint_shootout")

            try:
                if is_sprint:
                    fp1 = fastf1.get_session(year, event_name, "Practice 1")
                    fp1.load(laps=True, telemetry=False, weather=False, messages=False)
                    df = practiceExtractor(fp1, is_sprint=True)
                else:
                    fp1 = fastf1.get_session(year, event_name, "Practice 1")
                    fp2 = fastf1.get_session(year, event_name, "Practice 2")
                    fp3 = fastf1.get_session(year, event_name, "Practice 3")
                    fp1.load(laps=True, telemetry=False, weather=False, messages=False)
                    fp2.load(laps=True, telemetry=False, weather=False, messages=False)
                    fp3.load(laps=True, telemetry=False, weather=False, messages=False)
                    df = practiceExtractor(fp1, fp2, fp3, is_sprint=False)

                if df.empty:
                    log.warning("    ✗ %s: no usable practice data", event_name)
                    continue

                df["Year"]      = year
                df["EventName"] = event_name
                all_rows.append(df)
                log.info("    ✓ %-35s – %d drivers", event_name, len(df))

            except Exception as exc:
                log.warning("    ✗ %-35s – %s", event_name, str(exc)[:80])

    if not all_rows:
        log.warning("No practice features extracted.")
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(cache_path)
    log.info("Saved practice features (%s) to %s", result.shape, cache_path)
    return result


# ---------------------------------------------------------------------------
# Qualifying features
# ---------------------------------------------------------------------------

def load_qualifying_features(
    years: list[int] | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract Q1 / Q2 / Q3 lap times from the Qualifying session for all events.

    Matches the notebook pipeline: always loads 'Qualifying' session and
    computes only q3_delta and best_q_delta.

    Returns
    -------
    pd.DataFrame with columns: Abbreviation, Year, EventName, q3_delta, best_q_delta
    """
    years      = years      or TRAINING_YEARS
    cache_path = cache_path or QUALIFYING_CACHE

    if cache_path.exists() and not force:
        log.info("Loading qualifying features from cache: %s", cache_path)
        return pd.read_pickle(cache_path)

    log.info("Extracting qualifying features for years %s …", years)
    all_rows: list[pd.DataFrame] = []

    for year in years:
        log.info("  Season %d", year)
        schedule    = fastf1.get_event_schedule(year)
        race_events = schedule[schedule["EventFormat"] != "testing"]

        for _, event in race_events.iterrows():
            event_name = event["EventName"]

            try:
                session = fastf1.get_session(year, event_name, "Qualifying")
                session.load(laps=False, telemetry=False, weather=False, messages=False)
                results = session.results[["Abbreviation", "Q1", "Q2", "Q3"]].copy()
                results["Year"]      = year
                results["EventName"] = event_name
                # Convert timedeltas to seconds
                for col in ["Q1", "Q2", "Q3"]:
                    results[f"{col}_s"] = results[col].apply(
                        lambda v: v.total_seconds() if pd.notna(v) and hasattr(v, "total_seconds") else float("nan")
                    )
                results.drop(columns=["Q1", "Q2", "Q3"], inplace=True)
                all_rows.append(results)
                log.info("    ✓ %-35s – %d drivers", event_name, len(results))

            except Exception as exc:
                log.warning("    ✗ %-35s – %s", event_name, str(exc)[:80])

    if not all_rows:
        log.warning("No qualifying features extracted.")
        return pd.DataFrame()

    quali = pd.concat(all_rows, ignore_index=True)

    # Compute deltas matching the notebook
    pole_q3 = quali.groupby(["Year", "EventName"])["Q3_s"].transform("min")
    quali["q3_delta"] = quali["Q3_s"] - pole_q3

    quali["best_q_s"] = quali[["Q3_s", "Q2_s", "Q1_s"]].min(axis=1)
    pole_best_q = quali.groupby(["Year", "EventName"])["best_q_s"].transform("min")
    quali["best_q_delta"] = quali["best_q_s"] - pole_best_q

    result = quali[["Abbreviation", "Year", "EventName", "q3_delta", "best_q_delta"]]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(cache_path)
    log.info("Saved qualifying features (%s) to %s", result.shape, cache_path)
    return result


# ---------------------------------------------------------------------------
# Full dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(
    years: list[int] | None = None,
    test_year: int | None  = None,
    force_rebuild: bool    = False,
    rolling_windows: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the full feature-engineering pipeline and return train / test
    DataFrames that are ready for ``RaceRanker.fit()``.

    The DataFrames still contain metadata columns (EventName, Year, …) and
    the target column (Position).  Call ``get_drop_columns(df)`` from
    ``src.utils.helpers`` to obtain the list of columns to exclude when
    passing data to the model.

    Parameters
    ----------
    years          : Seasons to include.  Defaults to TRAINING_YEARS.
    test_year      : The hold-out season.  Defaults to TEST_YEAR.
    force_rebuild  : Re-run all feature extraction ignoring caches.
    rolling_windows: Override default rolling window sizes.

    Returns
    -------
    (train_data, test_data) : tuple of pd.DataFrame
    """
    years      = years      or TRAINING_YEARS
    test_year  = test_year  or TEST_YEAR
    rolling_windows = rolling_windows or ROLLING_WINDOWS

    # ------------------------------------------------------------------
    # Step 1: Raw race results
    # ------------------------------------------------------------------
    raw_results = load_raw_results(years=years, force=force_rebuild)

    # ------------------------------------------------------------------
    # Step 2: Historical features (driver / team / circuit)
    # ------------------------------------------------------------------
    if FEATURES_CACHE.exists() and not force_rebuild:
        log.info("Loading engineered features from cache: %s", FEATURES_CACHE)
        data = pd.read_pickle(FEATURES_CACHE)
    else:
        log.info("Computing historical features …")
        extractor = FeatureExtractor(raw_results, circuit_history=raw_results)
        data      = extractor.extract_all()

        # Step 3: Practice features
        practice = load_practice_features(years=years, force=force_rebuild)
        if not practice.empty:
            data = data.merge(
                practice,
                left_on =["Year", "EventName", "Abbreviation"],
                right_on=["Year", "EventName", "Driver"],
                how="left",
            ).drop(columns=["Driver"], errors="ignore")

        FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(FEATURES_CACHE)
        log.info("Saved feature cache (%s) to %s", data.shape, FEATURES_CACHE)

    # ------------------------------------------------------------------
    # Step 4: Rolling & championship context features
    # ------------------------------------------------------------------
    log.info("Adding rolling and championship context features …")
    engineer = FeatureEngineer(windows=rolling_windows)
    data     = engineer.fit_transform(data)

    # ------------------------------------------------------------------
    # Step 5: Drop rows with missing target
    # ------------------------------------------------------------------
    data = data[data["Position"].notna()].copy()
    data["Position"] = pd.to_numeric(data["Position"], errors="coerce")
    data = data[data["Position"].notna()].copy()
    log.info("Rows after dropping missing positions: %d", len(data))

    # ------------------------------------------------------------------
    # Step 6: Qualifying gap features (q3_delta, best_q_delta only)
    # ------------------------------------------------------------------
    quali = load_qualifying_features(years=years, force=force_rebuild)
    if not quali.empty:
        data = data.merge(
            quali,
            on=["Year", "EventName", "Abbreviation"],
            how="left",
        )

    # ------------------------------------------------------------------
    # Step 7: Sort (required for lambdarank grouping) and split
    # ------------------------------------------------------------------
    data = data.sort_values(["EventDate", "EventName"]).reset_index(drop=True)

    train = data[data["Year"] <  test_year].copy()
    test  = data[data["Year"] == test_year].copy()

    log.info("Train: %d rows  |  Test: %d rows", len(train), len(test))
    return train, test
