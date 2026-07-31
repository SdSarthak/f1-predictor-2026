# F1 Race Predictor 2026

A probabilistic Formula 1 race predictor built for the 2026 regulation reset.

It pulls historical results from the Ergast-compatible Jolpica API (optionally
enriched with FastF1 telemetry), engineers ~35 features, trains a gradient
boosted model on finishing position, blends the output with an Elo form rating
that updates race by race, applies 2026 regulation adjustments, and finally runs
a Monte Carlo race simulation to turn a single point estimate into win, podium
and DNF probabilities with confidence intervals.

---

## How it works

```
Jolpica/Ergast  ─┐
                 ├─► data pipeline ─► feature engineering ─► XGBoost / RF / ensemble
FastF1 (opt.)   ─┘                                                   │
                                                                     ▼
                          Elo form ratings ────────────► blended position estimate
                                                                     │
                                            2026 regulation adjustments (PU, aero, testing)
                                                                     │
                                                                     ▼
                                       Monte Carlo race simulation (safety cars, DNFs)
                                                                     │
                                                                     ▼
                                    win % / podium % / expected position / 90% CI
```

### 1. Data

| Source | Data | Notes |
|---|---|---|
| **Jolpica-F1** (Ergast mirror) | Results, qualifying, pit stops, standings, status | Paged 100 rows at a time; responses cached to `cache/` |
| **FastF1** | Lap times, tyre stints, weather, DRS telemetry | Optional — off by default because a cold cache takes tens of minutes per season |

Ergast shut down after 2024, so the client targets `https://api.jolpi.ca/ergast/f1`.
Override with the `F1_ERGAST_BASE_URL` environment variable or `data.ergast_base_url`
in the config.

### 2. Features

Rolling driver form and consistency, constructor form, grid-position features,
per-circuit history and overtaking difficulty, pit-stop efficiency, reliability,
rain skill, tyre degradation, Active Aero efficiency, engine/power-track
interactions and teammate head-to-head. Rolling and expanding features are
shifted by one race so a result never leaks into its own feature row.

### 3. Elo form layer

A modified Elo system rates drivers and constructors separately (30% driver /
70% car, matching how much of the variance the car explains). It updates after
every race via pairwise comparisons, with a larger K-factor for constructors so
car performance can move quickly. Ratings live in `models/elo_ratings.json`.
Its weight in the blend rises through the season (30% → 60%) as it accumulates
evidence.

### 4. 2026 regulation adjustments

- **Historical weight decay** — 2022-2025 team order matters less after a reset
- **New-engine uncertainty** — Audi and Red Bull Powertrains-Ford carry extra
  variance that decays over the first five races
- **Power unit profile** — 50/50 electric/ICE split, weighted by circuit type
- **Active Aero** — per-team ratings seeded from historical DRS effectiveness
- **Pre-season and opening-round evidence** — testing pace and reliability

### 5. Monte Carlo simulation

Each simulation samples a finishing position per driver from the predicted
position and its uncertainty, then plays out first-lap incidents, safety cars
(scaled by circuit type), red flags and per-lap reliability failures.

---

## Installation

```bash
git clone https://github.com/SdSarthak/f1-predictor-2026.git
cd f1-predictor-2026

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
```

No API keys are required. `.env.example` documents the one optional override.

---

## Usage

### Train

```bash
# Full pipeline: fetch, engineer, train, save to models/f1_predictor.joblib
python run_predictor.py --train

# Specific seasons and model type
python run_predictor.py --train --years 2023 2024 2025 --model ensemble

# Include FastF1 telemetry features (much slower - hours on a cold cache)
python run_predictor.py --train --telemetry
```

A cold `--train` over three seasons takes roughly 5-10 minutes: the Jolpica
mirror rate-limits, so the client sleeps between requests. Responses are cached
under `cache/`, so re-runs are fast.

### Predict

```bash
# Predict with an estimated grid
python run_predictor.py --predict --circuit Bahrain

# Predict a specific round with known qualifying results
python run_predictor.py --predict --year 2026 --round 5 --circuit Monaco \
    --grid russell=1 norris=2 leclerc=3

# Raw model output, without the Elo form blend
python run_predictor.py --predict --circuit Monza --no-elo

# Reproducible run - the same seed always gives the same probabilities
python run_predictor.py --predict --circuit Bahrain --seed 42
```

Predictions are printed and written to `predictions/<year>_R<round>_<circuit>.json`.

### Keep ratings current

```bash
# Fold a finished race into the Elo ratings
python run_predictor.py --update-race examples/australia_2026_results.csv \
    --circuit "Albert Park" --round 1

# Inspect current ratings
python run_predictor.py --ratings
```

The results CSV needs `driver_id`, `constructor_id` and `position`, plus optional
`quali_position` and `dnf` — see `examples/australia_2026_results.csv`.

### Other commands

```bash
python run_predictor.py --regulations           # 2026 regulation analysis
python run_predictor.py --fetch --years 2024 2025   # Fetch and cache data only
```

---

## Python API

```python
from src.predictor import F1Predictor

predictor = F1Predictor()

df = predictor.build_training_data(years=[2023, 2024, 2025])
metrics = predictor.train(df, model_type='xgboost')

results = predictor.predict_race(
    year=2026,
    round_num=1,
    circuit_name='Bahrain',
    grid_positions={'russell': 1, 'norris': 2},
    n_simulations=10000,
)

print(predictor.format_predictions(results))
predictor.save_predictions(results)
```

Individual components can be used on their own:

```python
from src.data.ergast_api import ErgastAPI
from src.features.feature_engineering import engineer_features
from src.models.race_updater import RaceByRaceUpdater
from src.simulation.monte_carlo import MonteCarloSimulator
from src.regulations.rules_2026 import Regulations2026

results = ErgastAPI().get_race_results(2024)
df, engineer = engineer_features(results, config={})

updater = RaceByRaceUpdater(elo_weight=0.4)
blended, uncertainty = updater.blend_predictions(ml_preds, ml_unc, driver_to_team)

mc = MonteCarloSimulator().run_monte_carlo(preds, unc, grid, reliability)

regs = Regulations2026()
regs.calculate_2026_team_uncertainty('sauber', race_number=1)
```

---

## Example output

```
==============================================================================
F1 2026 Race Prediction: Bahrain
Round 1 | Circuit Type: power_hungry | Elo weight: 40%
==============================================================================

Pos  Driver                            Grid     Win %   Podium %   Exp Pos  +/-Std
------------------------------------------------------------------------------
1    Kimi Antonelli (Mercedes)            2     30.5%      75.5%       2.9     3.1
2    George Russell (Mercedes)            1     42.2%      72.0%       3.0     3.1
3    Lewis Hamilton (Ferrari)             4      7.2%      43.8%       4.4     3.8
4    Lando Norris (McLaren)               5     10.5%      42.5%       4.8     4.0
5    Oscar Piastri (McLaren)              6      5.0%      35.0%       4.8     3.4

90% Confidence Intervals (top 10):
--------------------------------------------------
  Kimi Antonelli (Mercedes)                P1 - P6
  George Russell (Mercedes)                P1 - P7
  Lewis Hamilton (Ferrari)                 P1 - P11
```

---

## Configuration

Everything tunable lives in `config/settings.yaml`.

| Section | Controls |
|---|---|
| `data` | Training years, cache directory, API base URL, FastF1 toggle |
| `regulations_2026` | Weight decay, new-engine uncertainty |
| `track_categories` | High-speed / street / high-downforce / power-hungry circuits |
| `elo` | Enabled, blend weight, K-factors, ratings path |
| `model` | XGBoost and Random Forest hyperparameters, CV folds |
| `monte_carlo` | Simulation count, safety car / red flag / first-lap probabilities |
| `active_aero_ratings` | Per-team Active Aero baselines |
| `team_reliability_2026` | Per-team reliability used by the simulation |
| `rain_performance` | Per-driver wet-weather skill |

---

## Model performance

Measured on 2023-2025 (1,398 driver-races, 70 races), XGBoost with default
config. **Whole races are held out** — a random row split leaves 19 of a race's
20 cars in training while the 20th is scored, and finishing position is a
permutation within the race, so a row-level split flatters the model. The
scaler is fitted on the training rows only, inside each CV fold.

| Metric | Value |
|---|---|
| MAE | 3.29 positions |
| RMSE | 4.27 |
| R² | 0.45 |
| Within ±1 position | 21.1% |
| Within ±2 positions | 37.5% |
| Within ±3 positions | 55.0% |
| CV MAE (5-fold, grouped by race) | 3.06 ± 0.26 |

These are honest numbers from a leak-free feature set and a leak-free split.
Finishing position is substantially irreducible — retirements, strategy and
first-lap incidents are not predictable from pre-race information — which is
exactly why the output is reported as a probability distribution rather than a
single predicted order.

### Reproducibility

`--predict --seed N` fixes the Monte Carlo stage, so the same model, grid and
seed always produce the same probabilities. Without `--seed` the simulation is
freshly random each run.

Prediction also needs `data/training_data.parquet` alongside the model: the
per-driver feature rows and the fitted categorical encoders live in the
dataset, not in the `.joblib`. `--train` writes both. If the dataset is missing
the predictor warns and falls back to neutral feature values, which reduces the
output to little more than grid order.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover feature engineering (including leak-freedom), model training
and persistence, Monte Carlo statistical properties, the Elo system, 2026
regulation adjustments, Ergast pagination and the end-to-end prediction path.
No test touches the network.

---

## Project structure

```
f1-predictor-2026/
├── config/settings.yaml           # All tunable parameters
├── examples/                      # Sample race-result CSV
├── models/elo_ratings.json        # Persisted Elo ratings
├── src/
│   ├── constants.py               # Shared naming / circuit tables
│   ├── predictor.py               # Main prediction interface
│   ├── data/
│   │   ├── ergast_api.py          # Paged Jolpica/Ergast client
│   │   ├── fastf1_client.py       # FastF1 telemetry client
│   │   ├── pipeline.py            # Unified data pipeline
│   │   └── testing_data_2026.py   # 2026 pre-season and opening-round data
│   ├── features/feature_engineering.py
│   ├── models/
│   │   ├── trainer.py             # XGBoost / Random Forest / ensemble
│   │   ├── elo_updater.py         # Modified Elo rating system
│   │   └── race_updater.py        # ML + Elo blending
│   ├── simulation/monte_carlo.py
│   └── regulations/rules_2026.py
├── tests/
├── run_predictor.py               # CLI
└── requirements.txt
```

Generated artefacts (`cache/`, `data/`, `predictions/`, `*.joblib`) are gitignored.

---

## Requirements

Python 3.9+. See `requirements.txt`.

## Data sources

- Jolpica-F1 (Ergast mirror): https://api.jolpi.ca/ergast/f1
- FastF1: https://docs.fastf1.dev/

## License

MIT
