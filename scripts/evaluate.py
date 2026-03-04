#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import CACHE_DIR, MODELS_DIR, TEST_YEAR
from src.data.loaders import build_training_dataset
from src.models.ranker import RaceRanker
from src.utils.helpers import get_drop_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the trained F1 ranker.")
    p.add_argument("--year",       type=int,  default=TEST_YEAR,
                   help="Test season year (default: TEST_YEAR).")
    p.add_argument("--model",      type=str,  default=None,
                   help="Path to .pkl model (default: models/ranker.pkl).")
    p.add_argument("--save-plots", action="store_true",
                   help="Save plots to models/ instead of displaying them.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _position_error_distribution(errors: pd.Series) -> None:
    """Print a histogram-style summary of position errors."""
    print("\nPosition error distribution:")
    print(f"  MAE:    {errors.abs().mean():.3f}")
    print(f"  Median: {errors.abs().median():.3f}")
    print(f"  RMSE:   {np.sqrt((errors ** 2).mean()):.3f}")
    for threshold in (1, 2, 3, 5):
        pct = (errors.abs() <= threshold).mean() * 100
        print(f"  Within ±{threshold}: {pct:.1f} %")


def _winner_podium_classification(df: pd.DataFrame) -> None:
    """
    Evaluate whether the model correctly identifies race winners and podiums
    from its ranked predictions.
    """
    df = df.copy()
    df["pred_winner"] = (df["predicted_position"] == 1).astype(int)
    df["true_winner"] = (df["Position"] == 1).astype(int)
    df["pred_podium"] = (df["predicted_position"] <= 3).astype(int)
    df["true_podium"] = (df["Position"] <= 3).astype(int)

    for label, true_col, pred_col in [
        ("Winner (P1)",  "true_winner", "pred_winner"),
        ("Podium (P1–3)", "true_podium", "pred_podium"),
    ]:
        print(f"\n── {label} classification ──────────────────")
        print(classification_report(df[true_col], df[pred_col],
                                    target_names=["Not " + label.split()[0], label.split()[0]],
                                    zero_division=0))


def _plot_feature_importance(
    ranker: RaceRanker,
    top_n: int = 25,
    save_path: Path | None = None,
) -> None:
    imp = ranker.feature_importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color="steelblue")
    ax.set_title(f"LGBMRanker — Top {top_n} Feature Importances")
    ax.set_xlabel("Importance (split count)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("Saved feature importance plot → %s", save_path)
    else:
        plt.show()


def _plot_per_race_mae(
    report: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, max(5, len(report) * 0.4)))
    labels  = report["EventName"].str.replace(" Grand Prix", " GP")
    maes    = report["MAE"].values
    colors  = ["tomato" if m > report["MAE"].mean() else "steelblue" for m in maes]
    ax.barh(labels[::-1], maes[::-1], color=colors[::-1])
    ax.axvline(report["MAE"].mean(), linestyle="--", color="black",
               linewidth=1, label=f"Mean = {report['MAE'].mean():.2f}")
    ax.legend()
    ax.set_xlabel("MAE (positions)")
    ax.set_title(f"Per-Race MAE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        log.info("Saved per-race MAE plot → %s", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.Cache.offline_mode = True

    # ── Load model ────────────────────────────────────────────────────────
    model_path = Path(args.model) if args.model else MODELS_DIR / "ranker.pkl"
    if not model_path.exists():
        log.error("Model not found: %s  – run scripts/train.py first.", model_path)
        sys.exit(1)

    ranker: RaceRanker = RaceRanker.load(model_path)
    log.info("Loaded: %r", ranker)

    # ── Load test data ────────────────────────────────────────────────────
    log.info("Building test dataset for year %d …", args.year)
    _, test_data = build_training_dataset(test_year=args.year)

    drop_cols = get_drop_columns(test_data)
    X_test    = test_data.drop(columns=[c for c in drop_cols if c in test_data.columns])
    if ranker.feature_columns:
        X_test = X_test.reindex(columns=ranker.feature_columns)

    test_data = test_data.copy()
    test_data["predicted_position"] = ranker.predict_positions(
        X_test, test_data["EventName"]
    )
    test_data["position_error"] = test_data["predicted_position"] - test_data["Position"]

    # ── Summary statistics ────────────────────────────────────────────────
    overall_mae = test_data["position_error"].abs().mean()
    print(f"\n{'='*55}")
    print(f"  Evaluation  –  {args.year} season")
    print(f"{'='*55}")
    print(f"  Overall MAE:  {overall_mae:.3f} positions")

    _position_error_distribution(test_data["position_error"])
    _winner_podium_classification(test_data)

    # ── Per-race MAE ──────────────────────────────────────────────────────
    report = ranker.evaluation_report(test_data, drop_cols=drop_cols)
    print("\nPer-race MAE (all events):")
    print(report.to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────
    save_dir = MODELS_DIR if args.save_plots else None
    _plot_feature_importance(
        ranker,
        save_path=save_dir / "feature_importance.png" if save_dir else None,
    )
    _plot_per_race_mae(
        report,
        save_path=save_dir / "per_race_mae.png" if save_dir else None,
    )


if __name__ == "__main__":
    main()
