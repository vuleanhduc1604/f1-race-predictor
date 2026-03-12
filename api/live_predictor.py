"""
Live prediction: fetch qualifying + practice from FastF1 API, build features
from historical cache, and return predictions without using race results.

Used for 2026+ seasons where we want genuine pre-race predictions based only
on session pace data (no leakage from race outcomes).

Flow
----
1. Fetch qualifying + FP sessions online via FastF1.
2. Load historical race results from FEATURES_CACHE (rows before target date).
3. For each driver in the qualifying grid, compute historical features
   (career, team, circuit) with an efficient O(20 × n) lookup.
4. Compute rolling form + championship context from history.
5. Merge qualifying and practice features.
6. Predict using the trained model.
"""

from __future__ import annotations

import logging
import json as _json
import os
import sys
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CACHE_DIR,
    FEATURES_CACHE,
    MIN_CIRCUIT_RACES,
    ROLLING_WINDOWS,
    TEST_YEAR,
)
from src.data.extractors import practiceExtractor
from src.data.features import add_qualifying_features
from src.utils.helpers import get_drop_columns, is_dnf

from api.predictor import _load_model_meta, _load_ranker

log = logging.getLogger(__name__)

_FF1_HTTP_CACHE = "/tmp/fastf1_cache" if os.environ.get("VERCEL") else str(CACHE_DIR)


# ---------------------------------------------------------------------------
# FastF1 online / offline helpers
# ---------------------------------------------------------------------------

def _online():
    fastf1.Cache.offline_mode = False


def _offline():
    fastf1.Cache.offline_mode = True


# ---------------------------------------------------------------------------
# History loader
# ---------------------------------------------------------------------------

def _load_history(before_date: pd.Timestamp) -> pd.DataFrame:
    """Return all rows from FEATURES_CACHE with EventDate strictly before *before_date*."""
    if not FEATURES_CACHE.exists():
        raise FileNotFoundError(
            f"Features cache not found at {FEATURES_CACHE}. "
            "Run `python scripts/train.py` first to populate the cache."
        )
    df = pd.read_pickle(FEATURES_CACHE)
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    return df[df["EventDate"] < pd.to_datetime(before_date)].copy()


# ---------------------------------------------------------------------------
# Session fetcher
# ---------------------------------------------------------------------------

def _fetch_sessions(
    year: int, event: str
) -> tuple[object, object, object, object, bool, pd.Series]:
    """
    Go online temporarily and load qualifying + practice sessions.

    Returns
    -------
    (quali, fp1, fp2, fp3, is_sprint, event_info_series)
    Any session may be None if it could not be loaded.
    """
    _online()
    try:
        fastf1.Cache.enable_cache(_FF1_HTTP_CACHE)

        schedule = fastf1.get_event_schedule(year)
        mask = schedule["EventName"] == event
        if not mask.any():
            raise ValueError(f"Event '{event}' not found in {year} schedule.")
        event_info = schedule[mask].iloc[0]
        event_format = event_info.get("EventFormat", "conventional")
        is_sprint = event_format in ("sprint", "sprint_shootout")

        log.info("Event format: %s | is_sprint=%s", event_format, is_sprint)

        quali_name = "Sprint Qualifying" if is_sprint else "Qualifying"
        quali = None
        try:
            log.info("Loading %s session for %d %s …", quali_name, year, event)
            quali = fastf1.get_session(year, event, quali_name)
            quali.load(laps=True, telemetry=False, weather=False, messages=True)
            log.info("%s loaded — %d drivers in results", quali_name, len(quali.results))
        except Exception as exc:
            log.warning("Could not load %s: %s", quali_name, exc)

        fp1 = fp2 = fp3 = None
        for fp_name, var_name in [
            ("Practice 1", "fp1"),
            ("Practice 2", "fp2"),
            ("Practice 3", "fp3"),
        ]:
            if is_sprint and fp_name in ("Practice 2", "Practice 3"):
                continue
            try:
                log.info("Loading %s for %d %s …", fp_name, year, event)
                sess = fastf1.get_session(year, event, fp_name)
                sess.load(laps=True, telemetry=False, weather=False, messages=False)
                log.info("%s loaded — %d laps", fp_name, len(sess.laps))
                if var_name == "fp1":
                    fp1 = sess
                elif var_name == "fp2":
                    fp2 = sess
                else:
                    fp3 = sess
            except Exception as exc:
                log.warning("Could not load %s: %s", fp_name, exc)

        return quali, fp1, fp2, fp3, is_sprint, event_info

    finally:
        _offline()


# ---------------------------------------------------------------------------
# Manual grid position overrides
# Used for drivers who did not participate in qualifying but have a known
# starting grid position (e.g. grid penalties, late replacements).
# ---------------------------------------------------------------------------
_GRID_POSITION_OVERRIDES: dict[tuple[str, str], float] = {
    ("Australian Grand Prix", "VER"): 20.0,
    ("Australian Grand Prix", "SAI"): 21.0,
    ("Australian Grand Prix", "STR"): 22.0,
}

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def _to_float(value) -> float | None:
    """Convert a value to float, returning None for any NA-like value."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_driver_rows(
    quali,
    event_info: pd.Series,
    year: int,
    event: str,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one feature row per driver using qualifying results and historical
    statistics looked up from *history*.

    points_before_race is the sum of ALL historical race points (across all
    seasons), matching the notebook pipeline.
    """
    results = pd.DataFrame(quali.results)
    if results.empty:
        raise ValueError(f"Empty qualifying results for {year} {event}.")

    location = str(event_info.get("Location", ""))
    country = str(event_info.get("Country", ""))
    round_num = int(event_info.get("RoundNumber", 0))
    event_date = pd.to_datetime(event_info["EventDate"])

    has_status = "Status" in history.columns

    rows = []
    for _, dr in results.iterrows():
        abbr = dr.get("Abbreviation", "")
        if not abbr:
            continue

        team_name = str(dr.get("TeamName", ""))

        # History slices
        d_hist  = history[history["Abbreviation"] == abbr].sort_values("EventDate")
        t_hist  = history[history["TeamName"] == team_name].sort_values("EventDate") \
                  if team_name else pd.DataFrame()
        dt_hist = d_hist[d_hist["TeamName"] == team_name] if team_name else pd.DataFrame()
        c_hist  = history[history["Location"] == location].sort_values("EventDate") \
                  if location else pd.DataFrame()
        dc_hist = d_hist[d_hist["Location"] == location] if location else pd.DataFrame()

        n_d  = len(d_hist)
        n_t  = len(t_hist)
        n_dt = len(dt_hist)
        n_c  = len(c_hist)
        n_dc = len(dc_hist)

        row: dict = {
            "Year":         year,
            "EventName":    event,
            "EventDate":    event_date,
            "Location":     location,
            "Country":      country,
            "RoundNumber":  round_num,
            "Abbreviation": abbr,
            "DriverNumber": str(dr.get("DriverNumber", "")),
            "FirstName":    str(dr.get("FirstName", "")),
            "LastName":     str(dr.get("LastName", "")),
            "FullName":     str(dr.get("FullName", "")),
            "TeamName":     team_name,
            "TeamColor":    str(dr.get("TeamColor", "")),
            "GridPosition": _GRID_POSITION_OVERRIDES.get(
                (event, abbr), _to_float(dr.get("Position"))
            ),
            "DriverId":     abbr,
            "TeamId":       team_name,
        }

        # ── Driver career features ──────────────────────────────────────────
        if n_d > 0:
            d_is_dnf = d_hist["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
            row.update({
                "driver_total_races":            n_d,
                "driver_total_points_finishes":  int((d_hist["Points"] > 0).sum()),
                "driver_avg_finish_position":    float(d_hist["Position"].mean()),
                "driver_avg_grid_position":      float(d_hist["GridPosition"].mean()),
                "driver_avg_points_per_race":    float(d_hist["Points"].mean()),
                "driver_dnf_rate":               float(d_is_dnf.sum() / n_d) if has_status else None,
                "driver_avg_positions_gained":   float((d_hist["GridPosition"] - d_hist["Position"]).mean()),
                "driver_overtake_success_rate":  float((d_hist["Position"] < d_hist["GridPosition"]).sum() / n_d),
                "driver_finish_position_stddev": float(d_hist["Position"].std()),
            })
        else:
            row.update({
                "driver_total_races":            0,
                "driver_total_points_finishes":  0,
                "driver_avg_finish_position":    None,
                "driver_avg_grid_position":      None,
                "driver_avg_points_per_race":    0.0,
                "driver_dnf_rate":               None,
                "driver_avg_positions_gained":   None,
                "driver_overtake_success_rate":  None,
                "driver_finish_position_stddev": None,
            })

        # ── Team features ───────────────────────────────────────────────────
        if n_t > 0:
            t_is_dnf = t_hist["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
            row.update({
                "team_avg_finish_position":  float(t_hist["Position"].mean()),
                "team_avg_points_per_race":  float(t_hist["Points"].mean()),
                "team_total_podiums":        int((t_hist["Position"] <= 3).sum()),
                "team_dnf_rate":             float(t_is_dnf.sum() / n_t) if has_status else None,
                "team_recent_avg_position":  float(t_hist.tail(10)["Position"].mean() if n_t >= 10 else t_hist["Position"].mean()),
                "team_recent_total_points":  float(t_hist.tail(10)["Points"].sum() if n_t >= 10 else t_hist["Points"].sum()),
            })
        else:
            row.update({
                "team_avg_finish_position":  None,
                "team_avg_points_per_race":  0.0,
                "team_total_podiums":        0,
                "team_dnf_rate":             None,
                "team_recent_avg_position":  None,
                "team_recent_total_points":  0,
            })

        # ── Driver–team synergy ─────────────────────────────────────────────
        if n_dt > 0:
            dt_is_dnf = dt_hist["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
            row.update({
                "driver_team_avg_finish_position":   float(dt_hist["Position"].mean()),
                "driver_team_avg_points_per_race":   float(dt_hist["Points"].mean()),
                "driver_team_dnf_rate":              float(dt_is_dnf.sum() / n_dt) if has_status else None,
                "driver_team_avg_positions_gained":  float((dt_hist["GridPosition"] - dt_hist["Position"]).mean()),
                "driver_team_overtake_success_rate": float((dt_hist["Position"] < dt_hist["GridPosition"]).sum() / n_dt),
            })
        else:
            row.update({
                "driver_team_avg_finish_position":   None,
                "driver_team_avg_points_per_race":   0.0,
                "driver_team_dnf_rate":              None,
                "driver_team_avg_positions_gained":  None,
                "driver_team_overtake_success_rate": None,
            })

        # ── Circuit features ────────────────────────────────────────────────
        if n_c >= MIN_CIRCUIT_RACES:
            pos_chg   = (c_hist["GridPosition"] - c_hist["Position"]).abs()
            pole_rows = c_hist[c_hist["GridPosition"] == 1]
            top3_rows = c_hist[c_hist["GridPosition"] <= 3]
            c_is_dnf  = c_hist["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
            row.update({
                "circuit_avg_position_changes":      float(pos_chg.mean()),
                "circuit_pole_win_rate":             float((pole_rows["Position"] == 1).mean()) if len(pole_rows) else None,
                "circuit_top3_grid_podium_rate":     float((top3_rows["Position"] <= 3).mean()) if len(top3_rows) else None,
                "circuit_grid_position_correlation": float(c_hist["GridPosition"].corr(c_hist["Position"])),
                "circuit_avg_dnf_rate":              float(c_is_dnf.mean()) if has_status else None,
                "circuit_races_in_history":          int(c_hist["EventDate"].nunique()),
            })
        else:
            row.update({
                "circuit_avg_position_changes":      None,
                "circuit_pole_win_rate":             None,
                "circuit_top3_grid_podium_rate":     None,
                "circuit_grid_position_correlation": None,
                "circuit_avg_dnf_rate":              None,
                "circuit_races_in_history":          n_c,
            })

        # ── Driver–circuit specialist features ─────────────────────────────
        if n_dc > 0:
            dc_is_dnf = dc_hist["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
            row.update({
                "driver_circuit_avg_finish_position":   float(dc_hist["Position"].mean()),
                "driver_circuit_podiums":               int((dc_hist["Position"] <= 3).sum()),
                "driver_circuit_best_position":         float(dc_hist["Position"].min()),
                "driver_circuit_avg_grid_position":     float(dc_hist["GridPosition"].mean()),
                "driver_circuit_avg_positions_gained":  float((dc_hist["GridPosition"] - dc_hist["Position"]).mean()),
                "driver_circuit_overtake_success_rate": float((dc_hist["Position"] < dc_hist["GridPosition"]).sum() / n_dc),
                "driver_circuit_dnf_rate":              float(dc_is_dnf.sum() / n_dc) if has_status else None,
            })
        else:
            row.update({
                "driver_circuit_avg_finish_position":   None,
                "driver_circuit_podiums":               0,
                "driver_circuit_best_position":         None,
                "driver_circuit_avg_grid_position":     None,
                "driver_circuit_avg_positions_gained":  None,
                "driver_circuit_overtake_success_rate": None,
                "driver_circuit_dnf_rate":              None,
            })

        # ── Rolling form features ───────────────────────────────────────────
        for w in ROLLING_WINDOWS:
            last_w = d_hist.tail(w)
            if len(last_w) > 0:
                lw_is_dnf = last_w["Status"].apply(is_dnf) if has_status else pd.Series(dtype=float)
                row[f"driver_avg_positions_last_{w}"]       = float(last_w["Position"].mean())
                row[f"driver_total_points_last_{w}"]        = float(last_w["Points"].sum())
                row[f"driver_avg_dnf_rate_last_{w}"]        = float(lw_is_dnf.mean()) if has_status else None
                row[f"driver_podium_count_last_{w}"]        = int((last_w["Position"] <= 3).sum())
                row[f"driver_avg_position_gained_last_{w}"] = float((last_w["GridPosition"] - last_w["Position"]).mean())
            else:
                row[f"driver_avg_positions_last_{w}"]       = None
                row[f"driver_total_points_last_{w}"]        = None
                row[f"driver_avg_dnf_rate_last_{w}"]        = None
                row[f"driver_podium_count_last_{w}"]        = None
                row[f"driver_avg_position_gained_last_{w}"] = None

        # ── Championship context (current season only) ─────────────────────
        d_hist_season = d_hist[d_hist["Year"] == year]
        row["points_before_race"] = float(d_hist_season["Points"].sum()) if not d_hist_season.empty else 0.0

        # ── Drought counters ────────────────────────────────────────────────
        if n_d > 0:
            pos_series = d_hist["Position"]
            wins_at    = pos_series[pos_series == 1]
            podiums_at = pos_series[pos_series <= 3]
            pts_at     = d_hist[d_hist["Points"] > 0]

            def _races_since(subset_df: pd.DataFrame) -> int:
                if subset_df.empty:
                    return n_d
                return int((d_hist.index > subset_df.index[-1]).sum())

            row["races_since_last_win"]           = _races_since(wins_at)
            row["races_since_last_podium"]        = _races_since(podiums_at)
            row["races_since_last_points_finish"] = _races_since(pts_at)
        else:
            row["races_since_last_win"]           = 0
            row["races_since_last_podium"]        = 0
            row["races_since_last_points_finish"] = 0

        # race_number_in_season: use round number as proxy
        row["race_number_in_season"] = round_num

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure numeric dtypes on circuit/driver-circuit columns that may be object
    _NUMERIC_COLS = [
        "circuit_avg_position_changes", "circuit_pole_win_rate",
        "circuit_top3_grid_podium_rate", "circuit_grid_position_correlation",
        "circuit_avg_dnf_rate",
        "driver_circuit_avg_finish_position", "driver_circuit_avg_grid_position",
        "driver_circuit_avg_positions_gained", "driver_circuit_overtake_success_rate",
        "driver_circuit_dnf_rate",
    ]
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "GridPosition" in df.columns:
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")

    return df


def _add_championship_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive championship position / gap columns from points_before_race.
    Matches the notebook's FeatureEngineer.add_championship_context() logic
    applied to a single-event dataframe.
    """
    df = df.copy()
    pts = df["points_before_race"].fillna(0)
    df["championship_position_before_race"] = pts.rank(ascending=False, method="min")
    df["points_gap_to_leader_before_race"]  = pts.max() - pts
    pts_sorted = pts.sort_values(ascending=False)
    df["points_gap_to_next_before_race"] = pts_sorted.diff(-1).fillna(0).reindex(pts.index)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_live_prediction_stream(year: int, event: str):
    """
    Generator that yields SSE-formatted strings at each prediction step.

    Yields:
        ``event: progress\\ndata: {"message": "..."}\\n\\n``  for each step
        ``event: result\\ndata: {...}\\n\\n``                  for the final result
        ``event: fail\\ndata: {"detail": "..."}\\n\\n``        on error
    """
    def _emit(msg: str) -> str:
        return f"event: progress\ndata: {_json.dumps({'message': msg})}\n\n"

    try:
        log.info("Live prediction request: %d — %s", year, event)

        # ── 1. Fetch sessions ──────────────────────────────────────────────
        yield _emit("Fetching qualifying & practice sessions…")
        quali, fp1, fp2, fp3, is_sprint, event_info = _fetch_sessions(year, event)

        if quali is None:
            yield (
                f"event: fail\ndata: {_json.dumps({'detail': f'Qualifying session not available for {year} {event!r}. The session may not have taken place yet.'})}\n\n"
            )
            return

        session_names = ["Qualifying"] + [
            name for name, s in [("FP1", fp1), ("FP2", fp2), ("FP3", fp3)] if s is not None
        ]
        yield _emit(f"Sessions loaded: {', '.join(session_names)}.")

        event_date = pd.to_datetime(event_info["EventDate"])

        # ── 2. Load historical features ────────────────────────────────────
        yield _emit(f"Loading historical data (before {event_date.date()})…")
        history = _load_history(event_date)
        yield _emit(f"Historical data loaded ({len(history):,} rows).")

        # ── 3. Build per-driver feature rows ───────────────────────────────
        yield _emit("Computing per-driver historical features…")
        target_df = _build_driver_rows(quali, event_info, year, event, history)
        if target_df.empty:
            yield (
                f"event: fail\ndata: {_json.dumps({'detail': f'No drivers found in qualifying results for {year} {event!r}.'})}\n\n"
            )
            return
        yield _emit(f"Historical features built for {len(target_df)} drivers.")

        # ── 4. Championship rank features ──────────────────────────────────
        target_df = _add_championship_ranks(target_df)

        # ── 5. Qualifying gap features (q3_delta, best_q_delta) ────────────
        yield _emit("Adding qualifying gap features…")
        qr = pd.DataFrame(quali.results)

        def _td_to_s(series: pd.Series) -> pd.Series:
            return series.apply(
                lambda v: v.total_seconds()
                if pd.notna(v) and hasattr(v, "total_seconds")
                else np.nan
            )

        q_cols = {}
        for q_col in ("Q1", "Q2", "Q3"):
            q_cols[f"{q_col}_s"] = (
                _td_to_s(qr[q_col]) if q_col in qr.columns
                else pd.Series(np.nan, index=qr.index)
            )

        quali_df = pd.DataFrame({
            "Abbreviation": qr["Abbreviation"],
            "Year":         year,
            "EventName":    event,
            **q_cols,
        })
        target_df = add_qualifying_features(target_df, quali_df)

        # ── 6. Practice features ───────────────────────────────────────────
        if fp1 is not None:
            yield _emit("Extracting practice session features…")
            try:
                prac_df = (
                    practiceExtractor(fp1, is_sprint=True)
                    if is_sprint
                    else practiceExtractor(fp1, fp2, fp3, is_sprint=False)
                )
                if not prac_df.empty:
                    prac_df["Year"]      = year
                    prac_df["EventName"] = event
                    target_df = target_df.merge(
                        prac_df,
                        left_on  =["Abbreviation", "Year", "EventName"],
                        right_on =["Driver",        "Year", "EventName"],
                        how      ="left",
                    ).drop(columns=["Driver"], errors="ignore")
                    yield _emit("Practice features merged.")
            except Exception as exc:
                log.warning("Practice feature extraction failed: %s", exc)
                yield _emit(f"Practice features unavailable ({exc}); using historical features only.")
        else:
            yield _emit("No practice sessions available; using historical features only.")

        # ── 7. Load model and predict ──────────────────────────────────────
        yield _emit("Running LightGBM model…")
        _round_num = int(event_info.get("RoundNumber", 0)) or None
        ranker     = _load_ranker(year, round_number=_round_num)
        _meta      = _load_model_meta(year, round_number=_round_num)
        drop_cols = get_drop_columns(target_df)
        X         = target_df.drop(columns=[c for c in drop_cols if c in target_df.columns])
        if ranker.feature_columns:
            X = X.reindex(columns=ranker.feature_columns)

        target_df["predicted_position"] = ranker.predict_positions(X, target_df["EventName"])

        nan_counts = X.isna().sum()
        nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if not nan_features.empty:
            log.info("NaN feature counts across all drivers:\n%s", nan_features.to_string())

        yield _emit(f"Predictions complete for {len(target_df)} drivers.")

        # ── 7.5. Fetch actual race results if the race has finished ────────
        has_actuals = False
        median_error = None
        _online()
        try:
            fastf1.Cache.enable_cache(_FF1_HTTP_CACHE)
            race_sess = fastf1.get_session(year, event, "Race")
            race_sess.load(laps=False, telemetry=False, weather=False, messages=False)
            race_results = pd.DataFrame(race_sess.results)
            if not race_results.empty and "Position" in race_results.columns:
                race_results["Position"] = pd.to_numeric(race_results["Position"], errors="coerce")
                valid = race_results[race_results["Position"].notna()]
                if not valid.empty:
                    pos_map = valid.set_index("Abbreviation")["Position"].astype(int).to_dict()
                    target_df["actual_position"] = target_df["Abbreviation"].map(pos_map)
                    if target_df["actual_position"].notna().any():
                        target_df["error"] = (
                            target_df["predicted_position"] - target_df["actual_position"]
                        ).abs()
                        has_actuals = True
                        median_error = round(float(target_df["error"].median()), 3)
                        yield _emit(f"Actual race results loaded — Median Error: {median_error} positions.")
        except Exception as exc:
            log.info("Race results not yet available for %d '%s': %s", year, event, exc)
        finally:
            _offline()

        if not has_actuals:
            target_df["actual_position"] = None
            target_df["error"] = None

        # ── 8. Build response ──────────────────────────────────────────────
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
        for _, row in target_df.sort_values("predicted_position").iterrows():
            x_row = X.loc[row.name]
            features = {col: _serialize_val(x_row[col]) for col in feature_cols}
            drivers.append({
                "abbreviation":       str(row.get("Abbreviation", "")),
                "full_name":          f"{row.get('FirstName', '')} {row.get('LastName', '')}".strip(),
                "team":               str(row.get("TeamName", "")),
                "grid_position":      int(row["GridPosition"]) if pd.notna(row.get("GridPosition")) else None,
                "predicted_position": int(row["predicted_position"]),
                "actual_position":    int(row["actual_position"]) if has_actuals and pd.notna(row.get("actual_position")) else None,
                "error":              int(row["error"]) if has_actuals and pd.notna(row.get("error")) else None,
                "features":           features,
            })

        result = {
            "year":             year,
            "event":            event,
            "in_sample":        year <= TEST_YEAR,
            "has_actuals":      has_actuals,
            "median_error":     median_error,
            "feature_names":    feature_cols,
            "training_cutoff":  _meta.get("training_cutoff") if _meta else None,
            "drivers":          drivers,
            "source":           "live",
        }
        yield f"event: result\ndata: {_json.dumps(result)}\n\n"

    except ValueError as exc:
        yield f"event: fail\ndata: {_json.dumps({'detail': str(exc)})}\n\n"
    except Exception as exc:
        log.exception("Live prediction stream failed for %d '%s'", year, event)
        yield f"event: fail\ndata: {_json.dumps({'detail': str(exc)})}\n\n"


def run_live_prediction(year: int, event: str) -> dict:
    """
    Generate pre-race predictions for *event* in *year* using live session data.
    Returns the same dict shape as ``run_prediction``, with ``"source": "live"``.
    """
    for chunk in run_live_prediction_stream(year, event):
        if chunk.startswith("event: result\n"):
            return _json.loads(chunk.split("\ndata: ", 1)[1].strip())
        if chunk.startswith("event: fail\n"):
            detail = _json.loads(chunk.split("\ndata: ", 1)[1].strip())["detail"]
            raise ValueError(detail)
    raise ValueError("Prediction stream ended without a result.")
