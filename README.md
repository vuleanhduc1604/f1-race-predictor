# F1 Race Predictor

Predicts Formula 1 race finishing positions using LightGBM trained on FastF1 telemetry and timing data from the 2018–2025 seasons.

The default model is an **LGBMRegressor** (matching the original notebook baseline). A **LGBMRanker** (LambdaMART) mode is also available via `--mode ranker` for direct comparison.

---

## Features

- Extracts pace, long-run, and reliability features from FP1 / FP2 / FP3
- Extracts Q1 / Q2 / Q3 qualifying gaps from the qualifying session
- Engineers driver form, team form, circuit-specific records, and championship context using only past data (no leakage)
- Within-race relative features that normalise each driver's stats against the rest of the field
- Leave-one-year-out cross-validation for temporally honest hyperparameter tuning (Optuna)
- Two interchangeable model backends: `regressor` and `ranker`

---

## Results

Evaluated on the held-out **2025 season** (unseen during training). Lower MAE is better.

| Model | MAE (positions) |
|---|---|
| **LGBMRegressor** (default) | **2.033** |
| LGBMRanker (LambdaMART) | 2.042 |

Both models predict finishing positions to within ~2 places on average. The regressor marginally outperforms the ranker on this dataset, which is why it is the default.

---

## Project structure

```
f1-race-predictor/
├── cache/                          # FastF1 cache + pickled feature files
│   ├── practice_features.pkl       # Committed – extracted FP features
│   ├── qualifying_results.pkl      # Committed – Q1/Q2/Q3 times
│   └── data_with_all_features.pkl  # Committed – merged training dataset
├── models/                         # Saved .pkl model files (git-ignored)
├── notebooks/exploratory/          # Original development notebook
├── scripts/
│   ├── train.py                    # Train and save a model
│   ├── predict.py                  # Generate predictions for a race / season
│   └── evaluate.py                 # Evaluation report + plots
├── src/
│   ├── config.py                   # All constants, paths, hyperparameter defaults
│   ├── data/
│   │   ├── extractors.py           # practiceExtractor, qualifyingExtractor
│   │   ├── features.py             # FeatureExtractor, FeatureEngineer, helpers
│   │   └── loaders.py              # Cache-aware data loading + build_training_dataset()
│   ├── models/
│   │   ├── ranker.py               # RaceRanker (regressor / ranker modes)
│   │   └── tuning.py               # Optuna HPO with season-based CV
│   └── utils/
│       └── helpers.py              # is_dnf, get_drop_columns, get_race_groups, …
├── requirements.txt
└── .gitignore
```

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/vuleanhduc1604/f1-race-predictor.git
cd f1-race-predictor
pip install -r requirements.txt
```

---

## Quick start

### Using the committed feature caches (fastest)

The three processed feature caches are committed to the repo, so you can train immediately without fetching any data from FastF1.

```bash
# Train with default regressor params (no Optuna, uses committed caches)
python scripts/train.py --skip-tuning

# Evaluate on the 2025 hold-out season
python scripts/evaluate.py

# Predict the 2025 Australian Grand Prix
python scripts/predict.py --year 2025 --event "Australian Grand Prix"
```

### Full rebuild from FastF1

If you want to extend to newer races or re-extract everything from scratch, remove `--skip-tuning` and optionally pass `--force-rebuild`.

> **Note:** A full rebuild fetches data from the FastF1 API and may take 20–60 minutes depending on cache state and connection speed. Run once with `fastf1.Cache.offline_mode = False` (edit `scripts/train.py` line 70) to populate the local cache, then restore `True` for offline use.

```bash
python scripts/train.py --force-rebuild --n-trials 50
```

---

## Usage

### Train

```bash
# Default: LGBMRegressor, 50 Optuna trials
python scripts/train.py

# Skip Optuna — use the hardcoded default params from config.py
python scripts/train.py --skip-tuning

# Switch to LambdaMART ranker
python scripts/train.py --mode ranker --model-name ranker_lambdarank.pkl

# Save to a custom filename
python scripts/train.py --model-name my_model.pkl

# All options
python scripts/train.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `regressor` | `regressor` or `ranker` |
| `--n-trials` | `50` | Optuna HPO trials |
| `--skip-tuning` | off | Use hardcoded default params |
| `--force-rebuild` | off | Re-extract all features, ignore caches |
| `--test-year` | `2025` | Hold-out season for evaluation |
| `--model-name` | `ranker.pkl` | Output filename under `models/` |

### Predict

```bash
# All races in 2025
python scripts/predict.py --year 2025

# Single event
python scripts/predict.py --year 2025 --event "Australian Grand Prix"

# With a non-default model
python scripts/predict.py --model models/ranker_lambdarank.pkl --year 2025
```

### Evaluate

```bash
# Default: evaluates models/ranker.pkl on the 2025 season
python scripts/evaluate.py

# Save plots to models/
python scripts/evaluate.py --save-plots
```

Outputs:
- Overall MAE and position-error distribution
- Winner / podium classification report
- Per-race MAE table
- Feature importance plot
- Per-race MAE bar chart

---

## Model modes

| Mode | Class | Ranking direction | Groups needed |
|---|---|---|---|
| `regressor` (default) | `LGBMRegressor` | Ascending (lower predicted value = P1) | No |
| `ranker` | `LGBMRanker` (LambdaMART) | Descending (higher score = P1) | Yes |

To switch back to `ranker` permanently, change `DEFAULT_MODE = "ranker"` in `src/config.py`.

---

## Feature groups

| Group | Examples |
|---|---|
| Practice pace | `fp2_best_lap_delta`, `fp3_s1_delta`, `fp2_max_speed_i1` |
| Practice long run | `fp2_long_run_avg`, `fp2_long_run_consistency`, `fp2_avg_deg_rate` |
| Practice reliability | `fp1_total_laps`, `fp1_reliability_pct` |
| Qualifying | `q_best_gap`, `q3_gap_to_pole`, `reached_q3`, `reached_q2` |
| Driver form | `driver_avg_positions_last_5`, `driver_win_rate_last_10` |
| Team form | `team_avg_positions_last_5`, `team_win_rate_last_10` |
| Circuit history | `driver_circuit_avg_position`, `driver_circuit_wins` |
| Championship context | `driver_championship_position`, `points_gap_to_leader` |
| Within-race relative | `relative_driver_form_5`, `relative_team_form`, `relative_quali_gap` |
| Grid | `GridPosition` |

---

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Default model | LGBMRegressor | Matched or outperformed LGBMRanker on the 2025 hold-out in initial testing |
| Cross-validation | Leave-one-year-out | Respects temporal structure; random k-fold leaks future seasons into training |
| Qualifying features | `q_best_gap`, `q3_gap_to_pole` | Qualifying pace gap to pole is among the strongest pre-race predictors |
| Within-race relative features | `relative_driver_form_5`, etc. | Absolute form numbers are ambiguous without knowing who else is in the race |
| High-missing practice features | Dropped (>80% NaN) | Sprint-format weeks have no FP2/FP3; keeping them adds mostly noise |
| Leakage prevention | All features use `EventDate < current` | Prevents any future race data from influencing features for a given race |

---

## Cache files

| File | Size | Committed | Description |
|---|---|---|---|
| `cache/20XX/` | ~1–3 GB each | No | FastF1 per-session raw data |
| `cache/fastf1_http_cache.sqlite` | ~4.3 GB | No | FastF1 HTTP response cache |
| `cache/raw_results_2018_2025.pkl` | ~1.1 MB | No | Raw race results table (regenerated automatically) |
| `cache/practice_features.pkl` | ~1.3 MB | **Yes** | Extracted practice features per race |
| `cache/qualifying_results.pkl` | ~152 KB | **Yes** | Q1/Q2/Q3 times per driver per race |
| `cache/data_with_all_features.pkl` | ~3.4 MB | **Yes** | Fully merged training dataset |

To update the committed caches after adding new races, run:

```bash
python scripts/train.py --force-rebuild --skip-tuning
```

then commit the updated pkl files.

---

## Requirements

```
fastf1>=3.8
lightgbm>=4.0
optuna>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
tqdm
```

---

## Data source

Race data is sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) library, which accesses the official Formula 1 timing API.
