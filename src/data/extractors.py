"""
Session data extractors.

practiceExtractor  – Extracts pace, long-run, and reliability features from
                     FP1 / FP2 / FP3 (or single sprint FP1) sessions.

qualifyingExtractor – Extracts Q1 / Q2 / Q3 lap times in seconds from a
                      FastF1 qualifying session object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COMPOUND_MAP, MIN_DRIVERS, MIN_LAPS_PER_DRIVER


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_clean(session) -> pd.DataFrame:
    """Load laps from a FastF1 session, standardise compounds, drop outliers."""
    laps = pd.DataFrame(session.laps).copy()
    laps["Compound"] = laps["Compound"].map(COMPOUND_MAP).fillna(laps["Compound"])
    laps = laps[
        laps["PitOutTime"].isna()
        & laps["LapTime"].notna()
        & (laps["IsAccurate"] == True)
    ]
    if laps.empty:
        return laps
    laps["LapTime_s"]     = laps["LapTime"].dt.total_seconds()
    laps["Sector1Time_s"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Time_s"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Time_s"] = laps["Sector3Time"].dt.total_seconds()
    return laps


def _check_coverage(laps: pd.DataFrame) -> pd.DataFrame:
    """Drop drivers with too few laps and return empty DF if field is too small."""
    if laps.empty:
        return laps
    counts = laps.groupby("Driver")["LapNumber"].count()
    valid  = counts[counts >= MIN_LAPS_PER_DRIVER].index
    laps   = laps[laps["Driver"].isin(valid)]
    if laps["Driver"].nunique() < MIN_DRIVERS:
        return pd.DataFrame()
    return laps


def _get_long_run_features(laps: pd.DataFrame, prefix: str) -> dict:
    """Per-driver long-run average, consistency and tyre-degradation rate."""
    driver_avgs: dict[str, float] = {}
    features: dict[str, dict] = {}

    for driver, dlaps in laps.groupby("Driver"):
        long_run, deg_rates = [], []
        for _, stint in dlaps.groupby("Stint"):
            if len(stint) < 5:
                continue
            stint      = stint.sort_values("LapNumber").copy()
            best       = stint["LapTime_s"].min()
            stint      = stint[stint["LapTime_s"] <= best * 1.07]
            if len(stint) < 5:
                continue
            long_run.extend(stint["LapTime_s"].tolist())
            if len(stint) >= 7 and stint["TyreLife"].notna().all():
                slope = np.polyfit(stint["TyreLife"].values,
                                   stint["LapTime_s"].values, 1)[0]
                deg_rates.append(slope)

        avg = np.mean(long_run) if long_run else np.nan
        driver_avgs[driver] = avg
        features[driver] = {
            f"{prefix}_long_run_avg":         avg,
            f"{prefix}_long_run_consistency": np.std(long_run) if long_run else np.nan,
            f"{prefix}_avg_deg_rate":         np.mean(deg_rates) if deg_rates else np.nan,
        }

    valid_avgs = [v for v in driver_avgs.values() if not np.isnan(v)]
    field_best = min(valid_avgs) if valid_avgs else np.nan
    for driver, feat in features.items():
        avg = driver_avgs[driver]
        feat[f"{prefix}_long_run_delta"] = (
            avg - field_best if not np.isnan(avg) else np.nan
        )
    return features


def _get_pace_features(laps: pd.DataFrame, prefix: str) -> dict:
    """Best single-lap pace, sector deltas, trap speeds and compound-specific gaps."""
    session_best  = laps["LapTime_s"].min()
    s1_best_field = laps["Sector1Time_s"].min()
    s2_best_field = laps["Sector2Time_s"].min()
    s3_best_field = laps["Sector3Time_s"].min()

    compound_field_best = {
        c: laps.loc[laps["Compound"] == c, "LapTime_s"].min()
        if (laps["Compound"] == c).any() else np.nan
        for c in ("SOFT", "MEDIUM", "HARD")
    }

    features: dict[str, dict] = {}
    for driver, dlaps in laps.groupby("Driver"):
        best      = dlaps["LapTime_s"].min()
        delta     = best - session_best
        pct_off   = delta / session_best
        pb_laps   = dlaps[dlaps["IsPersonalBest"] == True]
        pb_delta  = pb_laps["LapTime_s"].min() - session_best if len(pb_laps) else delta

        s1 = dlaps[dlaps["Sector1Time_s"].notna()]
        s2 = dlaps[dlaps["Sector2Time_s"].notna()]
        s3 = dlaps[dlaps["Sector3Time_s"].notna()]

        compound_best = {
            f"{prefix}_best_{c.lower()}_delta": (
                dlaps.loc[dlaps["Compound"] == c, "LapTime_s"].min()
                - compound_field_best[c]
                if (dlaps["Compound"] == c).any()
                and not np.isnan(compound_field_best[c])
                else np.nan
            )
            for c in ("SOFT", "MEDIUM", "HARD")
        }

        features[driver] = {
            f"{prefix}_best_lap_delta":   delta,
            f"{prefix}_best_lap_pct_off": pct_off,
            f"{prefix}_pb_lap_delta":     pb_delta,
            f"{prefix}_s1_delta":         s1["Sector1Time_s"].min() - s1_best_field if len(s1) else np.nan,
            f"{prefix}_s2_delta":         s2["Sector2Time_s"].min() - s2_best_field if len(s2) else np.nan,
            f"{prefix}_s3_delta":         s3["Sector3Time_s"].min() - s3_best_field if len(s3) else np.nan,
            f"{prefix}_max_speed_i1":     dlaps["SpeedI1"].max(),
            f"{prefix}_max_speed_i2":     dlaps["SpeedI2"].max(),
            f"{prefix}_max_speed_st":     dlaps["SpeedST"].max(),
            f"{prefix}_max_speed_fl":     dlaps["SpeedFL"].max(),
            **compound_best,
        }
    return features


def _get_reliability_features(laps: pd.DataFrame, prefix: str) -> dict:
    """Lap count and reliability percentage for each driver."""
    max_laps = laps.groupby("Driver")["LapNumber"].count().max()
    return {
        driver: {
            f"{prefix}_total_laps":      len(dlaps),
            f"{prefix}_reliability_pct": len(dlaps) / max_laps if max_laps else np.nan,
        }
        for driver, dlaps in laps.groupby("Driver")
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def practiceExtractor(
    fp1_session,
    fp2_session=None,
    fp3_session=None,
    is_sprint: bool = False,
) -> pd.DataFrame:
    """
    Extract per-driver practice features from FP1 (and optionally FP2 / FP3).
    """
    fp1 = _check_coverage(_load_and_clean(fp1_session))
    fp2 = _check_coverage(_load_and_clean(fp2_session)) if fp2_session else pd.DataFrame()
    fp3 = _check_coverage(_load_and_clean(fp3_session)) if fp3_session else pd.DataFrame()

    all_drivers: set[str] = set()
    for laps in (fp1, fp2, fp3):
        if not laps.empty:
            all_drivers.update(laps["Driver"].unique())

    if not all_drivers:
        return pd.DataFrame()

    fp1_reliability = _get_reliability_features(fp1, "fp1") if not fp1.empty else {}
    fp1_long_run = _get_long_run_features(fp1, "fp1") if not fp1.empty else {}
    fp1_pace = _get_pace_features(fp1, "fp1") if (not fp1.empty and is_sprint) else {}
    fp2_pace = _get_pace_features(fp2, "fp2") if not fp2.empty else {}
    fp2_long_run = _get_long_run_features(fp2, "fp2") if not fp2.empty else {}
    fp3_pace = _get_pace_features(fp3, "fp3") if not fp3.empty else {}

    rows = []
    for driver in sorted(all_drivers):
        row: dict = {"Driver": driver}
        row.update(fp1_reliability.get(driver, {}))
        row.update(fp1_long_run.get(driver, {}))
        row.update(fp1_pace.get(driver, {}))
        row.update(fp2_pace.get(driver, {}))
        row.update(fp2_long_run.get(driver, {}))
        row.update(fp3_pace.get(driver, {}))
        row["has_sprint"] = int(is_sprint)
        rows.append(row)

    return pd.DataFrame(rows)


def qualifyingExtractor(quali_session) -> pd.DataFrame:
    """
    Extract Q1 / Q2 / Q3 lap times (seconds) from a FastF1 qualifying session.
    """
    results = pd.DataFrame(quali_session.results)
    if results.empty:
        return pd.DataFrame()

    def _to_seconds(td_series: pd.Series) -> pd.Series:
        return td_series.apply(
            lambda v: v.total_seconds() if pd.notna(v) and hasattr(v, "total_seconds") else np.nan
        )

    out = pd.DataFrame({
        "Driver": results["Abbreviation"],
        "Q1_s": _to_seconds(results["Q1"]),
        "Q2_s": _to_seconds(results["Q2"]),
        "Q3_s": _to_seconds(results["Q3"]),
    })
    return out.reset_index(drop=True)
