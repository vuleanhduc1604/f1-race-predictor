"""Utility / helper functions shared across the project."""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Race result helpers
# ---------------------------------------------------------------------------

_CLASSIFIED_STATUSES = frozenset(
    ["Finished", "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps",
     "+5 Laps", "+6 Laps", "+7 Laps", "+8 Laps", "+9 Laps", "Lapped"]
)


def is_dnf(status: str) -> bool:
    """Return True when *status* represents a Did Not Finish."""
    return status not in _CLASSIFIED_STATUSES


# ---------------------------------------------------------------------------
# Feature-column helpers
# ---------------------------------------------------------------------------

def get_drop_columns(df: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    """
    Return the subset of DROP_METADATA + DROP_LOW_IMPORTANCE + DROP_HIGH_MISSING
    that actually exist in *df*, plus any *extra* columns supplied by the caller.
    """
    from src.config import DROP_METADATA, DROP_LOW_IMPORTANCE, DROP_HIGH_MISSING, DROP_PREDICTIONS

    candidates = (
        DROP_METADATA
        + DROP_LOW_IMPORTANCE
        + DROP_HIGH_MISSING
        + DROP_PREDICTIONS
        + (extra or [])
    )
    return [c for c in candidates if c in df.columns]


def get_race_groups(df: pd.DataFrame) -> list[int]:
    """
    Return a list of group sizes (drivers per race) in the current row order.
    *df* must already be sorted by (EventDate, EventName) for lambdarank.
    """
    return df.groupby(["Year", "EventName"], sort=False).size().tolist()


def positions_to_relevance(positions: pd.Series, n_drivers: int = 20) -> pd.Series:
    """Convert 1-based integer positions to relevance labels (higher = better)."""
    return (n_drivers - positions).clip(lower=0).astype(int)
