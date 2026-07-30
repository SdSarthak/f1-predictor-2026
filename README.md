# F1 Race Predictor 2026

A machine learning-based Formula 1 race prediction system built for the 2026 regulation era.

## Overview

This predictor uses historical F1 data from multiple sources, advanced feature engineering, and Monte Carlo simulation to generate probabilistic race predictions. The system is specifically designed to handle the 2026 regulation reset with appropriate uncertainty modeling for new power units and aerodynamic changes.

---

## Features

### Data Sources
| Source | Data Type | Purpose |
|--------|-----------|---------|
| **Ergast API** | Race results, qualifying, pit stops, standings | Historical performance metrics |
| **FastF1** | Telemetry, lap times, tire data, weather | Detailed session analysis |

### Predictive Factors
| Factor | Description |
|--------|-------------|
| Grid Position | Historical qualifying-to-finish delta analysis |
| Tire Degradation | Lap time slope calculation per stint |
| Pit Stop Efficiency | Team average pit stop performance |
| Track Dynamics | Circuit categorization (high-speed, street, high-downforce) |
| Weather Impact | Track temperature and rain performance by driver |
| Reliability Score | DNF rates per engine manufacturer |
| Driver Form | Rolling 5-race performance average |
| Active Aero Efficiency | DRS-based estimation for 2026 movable wings |
| Power Unit Profile | 50/50 Electric/ICE performance modeling |

### 2026 Regulation Adjustments
- **Historical Weight Decay**: Reduced importance of 2023-2025 team dominance
- **New Engine Uncertainty**: Higher variance for Audi and Red Bull Powertrains-Ford
- **Active Aero Baseline**: Team ratings based on historical DRS effectiveness
- **Weight Reduction Effects**: 2022 adaptation scores as proxy for -30kg impact

---

## Project Structure

```
F1 Predictor/
├── config/
│   └── settings.yaml           # All configuration parameters
├── src/
│   ├── __init__.py
│   ├── predictor.py            # Main prediction interface
│   ├── data/
│   │   ├── ergast_api.py       # Ergast API client
│   │   ├── fastf1_client.py    # FastF1 telemetry client
│   │   └── pipeline.py         # Unified data pipeline
│   ├── features/
│   │   └── feature_engineering.py  # 30+ engineered features
│   ├── models/
│   │   └── trainer.py          # XGBoost/Random Forest training
│   ├── simulation/
│   │   └── monte_carlo.py      # Probabilistic race simulation
│   └── regulations/
│       └── rules_2026.py       # 2026-specific adjustments
├── run_predictor.py            # Command-line interface
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation

```bash
# Navigate to project directory
cd "F1 Predictor"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Command Line Interface

```bash
# Train model with 2022-2025 data
python run_predictor.py --train

# Train with specific years
python run_predictor.py --train --years 2023 2024 2025

# Train with different model type
python run_predictor.py --train --model ensemble

# Predict a race
python run_predictor.py --predict --circuit Bahrain

# Predict specific round
python run_predictor.py --predict --year 2026 --round 5 --circuit Monaco

# View 2026 regulation analysis
python run_predictor.py --regulations

# Fetch data only (without training)
python run_predictor.py --fetch --years 2024 2025
```

### Python API

```python
from src.predictor import F1Predictor

# Initialize
predictor = F1Predictor()

# Build training data
df = predictor.build_training_data(years=[2022, 2023, 2024, 2025])

# Train model
metrics = predictor.train(df, model_type='xgboost')

# Make predictions
results = predictor.predict_race(
    year=2026,
    round_num=1,
    circuit_name='Bahrain',
    n_simulations=10000
)

# Display results
print(predictor.format_predictions(results))

# Save predictions
predictor.save_predictions(results)
```

### Individual Components

```python
# Data fetching
from src.data.ergast_api import ErgastAPI
from src.data.fastf1_client import FastF1Client

ergast = ErgastAPI()
results = ergast.get_race_results(2024)
reliability = ergast.calculate_reliability_scores([2023, 2024])

fastf1 = FastF1Client()
laps = fastf1.get_lap_data(2024, 'Bahrain', 'R')
degradation = fastf1.calculate_tire_degradation(laps)

# Feature engineering
from src.features.feature_engineering import engineer_features
df_engineered, engineer = engineer_features(df)

# Model training
from src.models.trainer import train_model
trainer, metrics = train_model(df, feature_columns, model_type='xgboost')

# Monte Carlo simulation
from src.simulation.monte_carlo import MonteCarloSimulator
simulator = MonteCarloSimulator()
mc_results = simulator.run_monte_carlo(
    predicted_positions, uncertainties, grid, reliability
)

# 2026 regulations
from src.regulations.rules_2026 import Regulations2026
regs = Regulations2026()
uncertainty = regs.calculate_2026_team_uncertainty('red_bull', race_number=1)
```

---

## Output Format

```
======================================================================
F1 2026 Race Prediction: Bahrain
Round 1 | Circuit Type: power_hungry
======================================================================

Pos  Driver               Win %    Podium %    Exp Pos    ±Std
----------------------------------------------------------------------
  1  verstappen            34.2%      78.5%        2.1      1.8
  2  norris                22.1%      65.3%        2.8      2.1
  3  leclerc               18.7%      58.2%        3.2      2.3
  4  piastri               12.4%      42.1%        4.1      2.5
  5  sainz                  7.8%      35.6%        4.8      2.4
...

90% Confidence Intervals:
----------------------------------------
  verstappen: P1 - P5
  norris: P1 - P6
  leclerc: P1 - P7
```

---

## Configuration

Edit `config/settings.yaml` to customize:

| Section | Parameters |
|---------|------------|
| `data` | Training years, cache directory, API settings |
| `regulations_2026` | Weight decay, new engine uncertainty, Active Aero ratings |
| `track_categories` | High-speed, street circuit, power-hungry classifications |
| `model` | XGBoost/Random Forest hyperparameters, CV folds |
| `monte_carlo` | Simulation count, safety car probability, DNF rates |
| `rain_performance` | Driver wet weather skill ratings |

---

## Model Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| MAE | < 2.5 | Mean Absolute Error in positions |
| Within ±2 | > 65% | Predictions within 2 positions of actual |
| CV MAE | < 3.0 | Cross-validated Mean Absolute Error |

---

## Data Sources

- **Ergast Developer API**: http://ergast.com/mrd/
- **FastF1 Python Library**: https://docs.fastf1.dev/

---

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

Key dependencies:
- `fastf1` - F1 telemetry data
- `xgboost` - Gradient boosting model
- `scikit-learn` - Machine learning utilities
- `pandas`, `numpy` - Data processing
- `requests` - API calls

---

## License

MIT License
