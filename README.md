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

## Web UI

A full-stack web interface is included: a **FastAPI** backend wrapping the ML system, and a **React** frontend.

### Running locally

**1. Install Python dependencies (if you haven't already):**
```bash
pip install -r requirements.txt
```

**2. Start the API (from the project root):**
```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**3. Start the frontend (in a separate terminal):**
```bash
cd frontend
npm install   # first time only
npm run dev
```

The UI will be available at `http://localhost:5173`.

> **Note:** You must have a trained model at `models/ranker.pkl` before the API can serve predictions. Run `python scripts/train.py` first if you haven't already.

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/years` | List available season years |
| GET | `/events?year=2025` | List races for a season |
| GET | `/predict?year=2025&event=Australian Grand Prix` | Predict a race |
| GET | `/evaluate?year=2025` | Full-season evaluation metrics |
| GET | `/feature-importance?top_n=25` | Model feature importances |

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

### Grid

| Feature | Description |
|---|---|
| `GridPosition` | Starting grid position |

### Qualifying

| Feature | Description |
|---|---|
| `q_best_gap` | Driver's best qualifying time minus pole time (seconds) |
| `q3_gap_to_pole` | Q3 time minus pole time; NaN if driver did not reach Q3 |
| `reached_q3` | Binary: driver competed in Q3 |
| `reached_q2` | Binary: driver competed in Q2 |

### Practice pace (FP2 / FP3; FP1 for sprint weekends)

| Feature | Description |
|---|---|
| `fp2_best_lap_delta` / `fp3_best_lap_delta` | Best lap time minus session best (seconds) |
| `fp2_best_lap_pct_off` / `fp3_best_lap_pct_off` | Best lap delta as % of session best |
| `fp2_pb_lap_delta` / `fp3_pb_lap_delta` | Personal-best lap delta vs session best |
| `fp2_s1_delta` / `fp3_s1_delta` | Best sector 1 time minus field best |
| `fp2_s2_delta` / `fp3_s2_delta` | Best sector 2 time minus field best |
| `fp2_s3_delta` / `fp3_s3_delta` | Best sector 3 time minus field best |
| `fp2_max_speed_i1` / `fp3_max_speed_i1` | Max speed at speed trap 1 (km/h) |
| `fp2_max_speed_i2` / `fp3_max_speed_i2` | Max speed at speed trap 2 |
| `fp2_max_speed_st` / `fp3_max_speed_st` | Max speed at start/finish straight |
| `fp2_max_speed_fl` / `fp3_max_speed_fl` | Max speed at finish line |
| `fp2_best_soft_delta` / `fp3_best_soft_delta` | Best soft-tyre lap minus field best on softs |
| `fp2_best_medium_delta` / `fp3_best_medium_delta` | Best medium-tyre lap minus field best on mediums |

### Practice long run (FP1 / FP2)

| Feature | Description |
|---|---|
| `fp1_long_run_avg` / `fp2_long_run_avg` | Average lap time across qualifying long-run stints |
| `fp1_long_run_consistency` / `fp2_long_run_consistency` | Std deviation of long-run lap times |
| `fp1_avg_deg_rate` / `fp2_avg_deg_rate` | Tyre degradation rate (seconds/lap, from polyfit on tyre life) |
| `fp1_long_run_delta` / `fp2_long_run_delta` | Long-run average minus field best long-run average |

### Practice reliability (FP1)

| Feature | Description |
|---|---|
| `fp1_total_laps` | Total clean laps completed in FP1 |
| `fp1_reliability_pct` | Driver's lap count divided by the maximum in the session |

### Driver career history

| Feature | Description |
|---|---|
| `driver_total_races` | Total career races before this event |
| `driver_total_points_finishes` | Career number of points-scoring finishes |
| `driver_avg_finish_position` | Career average finishing position |
| `driver_avg_grid_position` | Career average starting grid position |
| `driver_avg_points_per_race` | Career average points scored per race |
| `driver_dnf_rate` | Career DNF rate |
| `driver_avg_positions_gained` | Career average grid-to-finish positions gained |
| `driver_overtake_success_rate` | Fraction of races where driver finished ahead of grid position |
| `driver_finish_position_stddev` | Std deviation of career finishing positions (consistency) |

### Driver recent form (rolling windows: 5 and 10 races)

| Feature | Description |
|---|---|
| `driver_avg_positions_last_5` / `_last_10` | Rolling average finishing position |
| `driver_total_points_last_5` / `_last_10` | Rolling total points scored |
| `driver_avg_dnf_rate_last_5` / `_last_10` | Rolling DNF rate |
| `driver_podium_count_last_5` / `_last_10` | Rolling podium count |
| `driver_avg_position_gained_last_5` / `_last_10` | Rolling average positions gained from grid |

### Team history

| Feature | Description |
|---|---|
| `team_avg_finish_position` | All-time average finish position for the team |
| `team_avg_points_per_race` | All-time average points per team entry |
| `team_total_podiums` | All-time podium count |
| `team_dnf_rate` | All-time DNF rate |
| `team_recent_avg_position` | Average position over last 10 team entries |
| `team_recent_total_points` | Total points over last 10 team entries |

### Driver–team synergy

| Feature | Description |
|---|---|
| `driver_team_avg_finish_position` | Average finish position for this driver in this team |
| `driver_team_avg_points_per_race` | Average points in this driver–team combination |
| `driver_team_dnf_rate` | DNF rate in this driver–team combination |
| `driver_team_avg_positions_gained` | Average positions gained in this driver–team combination |
| `driver_team_overtake_success_rate` | Overtake rate in this driver–team combination |

### Circuit characteristics

| Feature | Description |
|---|---|
| `circuit_avg_position_changes` | Historical average grid-to-finish position change at this circuit |
| `circuit_pole_win_rate` | Historical rate of pole position converting to win |
| `circuit_top3_grid_podium_rate` | Historical rate of top-3 grid positions finishing on the podium |
| `circuit_grid_position_correlation` | Correlation between grid and finish position (overtaking difficulty) |
| `circuit_avg_dnf_rate` | Historical DNF rate at this circuit |
| `circuit_races_in_history` | Number of historical races used to compute circuit stats |

### Driver circuit specialist

| Feature | Description |
|---|---|
| `driver_circuit_avg_finish_position` | Driver's average finishing position at this circuit |
| `driver_circuit_podiums` | Driver's total podiums at this circuit |
| `driver_circuit_best_position` | Driver's best ever finish at this circuit |
| `driver_circuit_avg_grid_position` | Driver's average starting position at this circuit |
| `driver_circuit_avg_positions_gained` | Driver's average positions gained at this circuit |
| `driver_circuit_overtake_success_rate` | Driver's overtake success rate at this circuit |
| `driver_circuit_dnf_rate` | Driver's DNF rate at this circuit |

### Championship context

| Feature | Description |
|---|---|
| `points_before_race` | Driver's cumulative championship points before this race |
| `championship_position_before_race` | Driver's championship standing before this race |
| `points_gap_to_leader_before_race` | Points behind the championship leader |
| `points_gap_to_next_before_race` | Points behind the driver directly ahead in the standings |
| `races_since_last_win` | Races elapsed since driver's last victory |
| `races_since_last_podium` | Races elapsed since driver's last podium |
| `races_since_last_points_finish` | Races elapsed since driver last scored points |
| `race_number_in_season` | Sequential race number in the current season |

### Within-race relative features

| Feature | Source feature normalised |
|---|---|
| `relative_driver_form_5` | `driver_avg_positions_last_5` minus race-field median |
| `relative_driver_form_10` | `driver_avg_positions_last_10` minus race-field median |
| `relative_team_form` | `team_recent_avg_position` minus race-field median |
| `relative_driver_points_5` | `driver_total_points_last_5` minus race-field median |
| `relative_driver_points_10` | `driver_total_points_last_10` minus race-field median |
| `relative_champ_pos` | `championship_position_before_race` minus race-field median |

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
