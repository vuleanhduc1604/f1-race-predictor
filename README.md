# F1 Race Predictor

Predicts Formula 1 race finishing positions using LightGBM trained on FastF1 telemetry and timing data from the 2018–2026 seasons.

**Live demo:** [f1-race-predictor-bay.vercel.app](https://f1-race-predictor-bay.vercel.app/)

The default model is an **LGBMRegressor** (matching the original notebook baseline). A **LGBMRanker** (LambdaMART) mode is also available via `--mode ranker` for direct comparison.

---

## Features

- Extracts pace, long-run, and reliability features from FP1 / FP2 / FP3
- Extracts Q1 / Q2 / Q3 qualifying gaps from the qualifying session
- Engineers driver form, team form, circuit-specific records, and championship context using only past data (no leakage)
- Within-race relative features that normalise each driver's stats against the rest of the field
- Leave-one-year-out cross-validation for temporally honest hyperparameter tuning (Optuna)
- Two interchangeable model backends: `regressor` and `ranker`
- **Post-race pipeline** that appends new race results and retrains a round-specific model after each race
- **Round-level train/test split** so the model for any race is trained on all prior races only
- **Live SSE streaming** with a step-by-step progress indicator in the UI for 2026+ predictions

---

## Results

Evaluated on the held-out **2025 season** (model trained on 2018–2024 at the time of evaluation). Lower median error is better.

| Model | Median error (positions) |
|---|---|
| **LGBMRegressor** (default) | **2.033** |
| LGBMRanker (LambdaMART) | 2.042 |

Both models predict finishing positions to within ~2 places on average. The regressor marginally outperforms the ranker on this dataset, which is why it is the default.

> **Current model:** Retrained on the full 2018–2026 dataset. 2026 races not yet in training appear as live out-of-sample predictions.

---

## Web UI

A full-stack web interface is included: a **FastAPI** backend wrapping the ML system, and a **React** frontend.

**Live:** [f1-race-predictor-bay.vercel.app](https://f1-race-predictor-bay.vercel.app/) (frontend on Vercel, backend on Railway)

### Running locally

**1. Install Python dependencies:**
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

> **Note:** You must have a trained model under `models/` before the API can serve predictions. Run `python scripts/train.py` first if you haven't already.

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/years` | List available season years |
| GET | `/events?year=2026` | List races for a season |
| GET | `/predict?year=2026&event=Australian Grand Prix` | Predict a race (historical) |
| GET | `/predict-live?year=2026&event=Chinese Grand Prix` | Predict a race using live FastF1 data (SSE stream) |
| GET | `/evaluate?year=2025` | Full-season evaluation metrics |
| GET | `/feature-importance?top_n=25` | Model feature importances |

---

## Post-race pipeline

After each race finishes, run `scripts/post_race.py` to:

1. Append the completed race's results to `cache/raw_results_2018_2026.pkl`
2. Delete derived caches so they rebuild from scratch on the next training run
3. Train a round-specific model (e.g. `ranker_2026_R02.pkl`) on all races before the next round
4. Save a JSON metadata sidecar tracking the training cutoff

```bash
# After Australian GP (Round 1) — trains ranker_2026_R02.pkl for Chinese GP
python scripts/post_race.py --year 2026 --completed-round 1

# Or reference by name
python scripts/post_race.py --year 2026 --completed-event "Australian Grand Prix"
```

| Flag | Description |
|---|---|
| `--year` | Season year (required) |
| `--completed-round` | Round number just finished (mutually exclusive with `--completed-event`) |
| `--completed-event` | Event name just finished (mutually exclusive with `--completed-round`) |
| `--next-round` | Override which round to target (default: completed + 1) |
| `--mode` | `regressor` or `ranker` (default: `regressor`) |
| `--skip-cache-refresh` | Skip fetching results, only retrain |
| `--force` | Overwrite existing model for this round |

### Model naming convention

| File | Trained on | Used for |
|---|---|---|
| `ranker_2026.pkl` | 2018–2025 | Baseline; used when no round-specific model exists |
| `ranker_2026_R02.pkl` | 2018–2025 + Australian GP 2026 | Chinese GP 2026 (Round 2) |
| `ranker_2026_R03.pkl` | 2018–2025 + Rounds 1–2 | Round 3 onward |

### Model resolution

When predicting a race, `_load_ranker` resolves the model in this order:

1. `ranker_{year}_R{round:02d}.pkl` — round-specific (post-race pipeline)
2. `ranker_{year}.pkl` — year-specific baseline
3. `ranker_2026.pkl` — general 2026 fallback
4. `ranker.pkl` — last-resort fallback

### Training metadata

Each model has a JSON sidecar (e.g. `ranker_2026_R02_meta.json`) that records the training cutoff. The frontend banner reads this to show:

> *Model trained through **Australian Grand Prix 2026** (Round 1).*

---

## Project structure

```
f1-race-predictor/
├── cache/
│   ├── raw_results_2018_2026.pkl   # Source of truth — all race results
│   └── data_with_all_features.pkl  # Merged feature dataset (used by live predictor)
├── models/
│   ├── ranker_2026.pkl             # Baseline model (2018–2025)
│   ├── ranker_2026_R02.pkl         # Round-specific model (example)
│   └── ranker_2026_R02_meta.json   # Training metadata sidecar
├── scripts/
│   ├── train.py                    # Train and save a model
│   ├── post_race.py                # Post-race pipeline: update cache + retrain
│   ├── predict.py                  # Generate predictions for a race / season
│   └── evaluate.py                 # Evaluation report + plots
├── api/
│   ├── main.py                     # FastAPI app
│   ├── predictor.py                # Prediction logic, model loading
│   └── live_predictor.py           # Live SSE prediction stream (2026+)
├── src/
│   ├── config.py                   # All constants, paths, hyperparameter defaults
│   ├── data/
│   │   ├── extractors.py           # practiceExtractor, qualifyingExtractor
│   │   ├── features.py             # FeatureExtractor, FeatureEngineer, helpers
│   │   └── loaders.py              # Cache-aware loading + build_training_dataset()
│   ├── models/
│   │   ├── ranker.py               # RaceRanker (regressor / ranker modes)
│   │   └── tuning.py               # Optuna HPO with season-based CV
│   └── utils/
│       └── helpers.py              # is_dnf, get_drop_columns, get_race_groups, …
├── frontend/                       # React + Vite frontend
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

Two cache files are committed to the repo so you can train immediately without fetching anything from FastF1.

```bash
# Train with default regressor params (no Optuna)
python scripts/train.py --skip-tuning

# Evaluate on the 2026 hold-out season
python scripts/evaluate.py

# Predict the 2026 Australian Grand Prix
python scripts/predict.py --year 2026 --event "Australian Grand Prix"
```

### Full rebuild from FastF1

```bash
python scripts/train.py --force-rebuild --n-trials 50
```

> **Note:** A full rebuild fetches data from the FastF1 API and may take 20–60 minutes depending on cache state and connection speed.

---

## Usage

### Train

```bash
# Default: LGBMRegressor, 50 Optuna trials
python scripts/train.py

# Skip Optuna — use hardcoded default params from config.py
python scripts/train.py --skip-tuning

# Switch to LambdaMART ranker
python scripts/train.py --mode ranker --model-name ranker_lambdarank.pkl

# Save to a custom filename
python scripts/train.py --model-name my_model.pkl
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `regressor` | `regressor` or `ranker` |
| `--n-trials` | `50` | Optuna HPO trials |
| `--skip-tuning` | off | Use hardcoded default params |
| `--force-rebuild` | off | Re-extract all features, ignore caches |
| `--test-year` | `2026` | Hold-out season for evaluation |
| `--model-name` | `ranker_2026.pkl` | Output filename under `models/` |

### Predict

```bash
# All races in 2026
python scripts/predict.py --year 2026

# Single event
python scripts/predict.py --year 2026 --event "Australian Grand Prix"

# With a non-default model
python scripts/predict.py --model models/ranker_lambdarank.pkl --year 2026
```

### Evaluate

```bash
# Default: evaluates on the 2026 season
python scripts/evaluate.py

# Save plots to models/
python scripts/evaluate.py --save-plots
```

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
| Round-level split | `EventDate < target_round_date` | Ensures the model for each race is trained only on races that happened before it |

---

## Cache files

Only two cache files need to be present:

| File | Description |
|---|---|
| `cache/raw_results_2018_2026.pkl` | Source of truth — all race results; updated by `post_race.py` after each race |
| `cache/data_with_all_features.pkl` | Fully merged feature dataset; used by the live predictor for fast history lookups |

All other derived caches (`practice_features.pkl`, `qualifying_results.pkl`) are intermediate build artefacts and are deleted by `post_race.py` after each update. They are rebuilt automatically from local FastF1 session files on the next training run.

---

## Requirements

**Inference / API** (`requirements.txt`):
```
fastf1>=3.8
lightgbm>=4.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
fastapi>=0.110
uvicorn[standard]>=0.27
```

**Training / development** (`requirements-dev.txt`, includes all of the above plus):
```
optuna>=3.0
matplotlib>=3.7
tqdm
```

---

## Data source

Race data is sourced from the [FastF1](https://github.com/theOehrly/Fast-F1) library, which accesses the official Formula 1 timing API.
