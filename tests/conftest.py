"""Shared fixtures. Also puts the repo root on `sys.path` for `import src`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = str(ROOT / "config" / "settings.yaml")

CONSTRUCTORS = [
    ('red_bull', 'Red Bull Racing', ['verstappen', 'perez']),
    ('mercedes', 'Mercedes', ['russell', 'hamilton']),
    ('ferrari', 'Ferrari', ['leclerc', 'sainz']),
    ('mclaren', 'McLaren', ['norris', 'piastri']),
    ('haas', 'Haas', ['hulkenberg', 'magnussen']),
]

CIRCUITS = [
    ('bahrain', 'Bahrain International Circuit'),
    ('monza', 'Autodromo Nazionale di Monza'),
    ('monaco', 'Circuit de Monaco'),
    ('silverstone', 'Silverstone Circuit'),
]


@pytest.fixture(scope="session")
def config_path():
    """Path to the real project config, so tests exercise the shipped values."""
    return CONFIG_PATH


@pytest.fixture(scope="session")
def raw_results() -> pd.DataFrame:
    """
    A synthetic but structurally faithful stand-in for the pipeline output.

    Two seasons, four circuits, ten drivers. Finishing order is driven by a
    per-driver strength plus noise, so learned features are meaningful.
    """
    rng = np.random.default_rng(20260731)

    drivers = [(driver, cid, cname)
               for cid, cname, names in CONSTRUCTORS
               for driver in names]
    strength = {driver: i for i, (driver, _, _) in enumerate(drivers)}

    rows = []
    for year in (2024, 2025):
        for round_num, (circuit_id, circuit_name) in enumerate(CIRCUITS, start=1):
            noise = {driver: strength[driver] + rng.normal(0, 2) for driver, _, _ in drivers}
            order = sorted(noise, key=noise.get)
            grid_noise = {driver: strength[driver] + rng.normal(0, 1.5) for driver, _, _ in drivers}
            grid_order = sorted(grid_noise, key=grid_noise.get)

            for driver, constructor_id, constructor_name in drivers:
                finish = order.index(driver) + 1
                rows.append({
                    'year': year,
                    'round': round_num,
                    'race_name': circuit_name,
                    'circuit_id': circuit_id,
                    'circuit_name': circuit_name,
                    'date': f"{year}-0{round_num}-01",
                    'driver_id': driver,
                    'driver_code': driver[:3].upper(),
                    'driver_name': driver.title(),
                    'constructor_id': constructor_id,
                    'constructor_name': constructor_name,
                    'grid_position': grid_order.index(driver) + 1,
                    'finish_position': finish,
                    'points': max(0, 11 - finish),
                    'status': 'Finished' if finish < 10 else 'Engine',
                    'laps_completed': 57,
                    'reliability_score': 0.9,
                    'avg_pit_time': 22.5 + rng.normal(0, 0.5),
                    'team_avg_pit_time': 22.5,
                    'avg_track_temp': 32.0,
                    'is_wet_race': False,
                    'active_aero_efficiency': 0.85,
                    'historical_weight': 0.3,
                    'is_high_speed': int(circuit_id in ('monza',)),
                    'is_high_downforce': int(circuit_id in ('monaco',)),
                    'is_street_circuit': int(circuit_id in ('monaco',)),
                    'is_power_hungry': int(circuit_id in ('monza', 'bahrain', 'silverstone')),
                })

    df = pd.DataFrame(rows)
    df['position_delta'] = df['grid_position'] - df['finish_position']
    return df


@pytest.fixture
def engineered(raw_results):
    """Feature-engineered training frame plus the fitted FeatureEngineer."""
    import yaml

    from src.features.feature_engineering import engineer_features

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return engineer_features(raw_results.copy(), config)
