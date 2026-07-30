"""Data pipeline merges, using stubbed source data (no network)."""

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import F1DataPipeline


@pytest.fixture
def pipeline(config_path):
    return F1DataPipeline(config_path)


@pytest.fixture
def base_results():
    return pd.DataFrame([
        {'year': 2024, 'round': 1, 'race_name': 'Bahrain Grand Prix',
         'circuit_id': 'bahrain', 'circuit_name': 'Bahrain International Circuit',
         'driver_id': 'russell', 'driver_code': 'RUS', 'constructor_id': 'mercedes',
         'constructor_name': 'Mercedes', 'grid_position': 1, 'finish_position': 1.0,
         'points': 25.0, 'status': 'Finished'},
        {'year': 2024, 'round': 1, 'race_name': 'Bahrain Grand Prix',
         'circuit_id': 'bahrain', 'circuit_name': 'Bahrain International Circuit',
         'driver_id': 'verstappen', 'driver_code': 'VER', 'constructor_id': 'red_bull',
         'constructor_name': 'Red Bull', 'grid_position': 2, 'finish_position': None,
         'points': 0.0, 'status': 'Engine'},
    ])


def test_qualifying_merge_adds_the_column(pipeline, base_results):
    quali = pd.DataFrame([
        {'year': 2024, 'round': 1, 'driver_id': 'russell', 'quali_position': 1},
        {'year': 2024, 'round': 1, 'driver_id': 'verstappen', 'quali_position': 2},
    ])

    merged = pipeline._merge_qualifying(base_results, quali)

    assert len(merged) == len(base_results)
    assert merged.set_index('driver_id').loc['verstappen', 'quali_position'] == 2


def test_qualifying_merge_is_a_no_op_when_empty(pipeline, base_results):
    merged = pipeline._merge_qualifying(base_results, pd.DataFrame())

    assert merged.equals(base_results)


def test_pit_stop_merge_produces_driver_and_team_aggregates(pipeline, base_results):
    stops = pd.DataFrame([
        {'year': 2024, 'round': 1, 'driver_id': 'russell', 'duration_seconds': 22.0},
        {'year': 2024, 'round': 1, 'driver_id': 'russell', 'duration_seconds': 24.0},
        {'year': 2024, 'round': 1, 'driver_id': 'verstappen', 'duration_seconds': 21.0},
    ])

    merged = pipeline._merge_pit_stops(base_results, stops).set_index('driver_id')

    assert merged.loc['russell', 'avg_pit_time'] == pytest.approx(23.0)
    assert merged.loc['russell', 'num_pit_stops'] == 2
    assert merged.loc['russell', 'best_pit_time'] == pytest.approx(22.0)


def test_pit_stop_merge_leaves_placeholders_when_empty(pipeline, base_results):
    merged = pipeline._merge_pit_stops(base_results, pd.DataFrame())

    assert merged['avg_pit_time'].isna().all()
    assert merged['num_pit_stops'].isna().all()


def test_reliability_merge_defaults_missing_teams(pipeline, base_results):
    reliability = pd.DataFrame([{'constructor_id': 'mercedes', 'reliability_score': 0.97}])

    merged = pipeline._merge_reliability(base_results, reliability).set_index('driver_id')

    assert merged.loc['russell', 'reliability_score'] == pytest.approx(0.97)
    assert merged.loc['verstappen', 'reliability_score'] == pytest.approx(0.9)


def test_tire_degradation_placeholders_keep_the_schema_stable(pipeline, base_results):
    """Without telemetry these columns must still exist for the model."""
    merged = pipeline._merge_tire_degradation(base_results, pd.DataFrame())

    for column in ('avg_deg_per_lap_pct', 'avg_deg_slope', 'season_avg_deg'):
        assert column in merged.columns
        assert merged[column].isna().all()


def test_active_aero_falls_back_to_config_ratings(pipeline, base_results):
    """The empty-telemetry path must populate the feature the model consumes."""
    merged = pipeline._merge_drs_efficiency(base_results, pd.DataFrame()).set_index('driver_id')

    assert 'active_aero_efficiency' in merged.columns
    assert merged.loc['russell', 'active_aero_efficiency'] == pytest.approx(0.98)
    # `red_bull` resolves via the display-name map even though the raw Ergast
    # constructor_name is just "Red Bull".
    assert merged.loc['verstappen', 'active_aero_efficiency'] == pytest.approx(0.85)


def test_active_aero_is_normalised_from_telemetry(pipeline, base_results):
    drs = pd.DataFrame([
        {'Year': 2024, 'Team': 'Mercedes', 'DRSSpeedGain': 20.0, 'DRSUsagePct': 15.0},
        {'Year': 2024, 'Team': 'Red Bull', 'DRSSpeedGain': 10.0, 'DRSUsagePct': 12.0},
    ])

    merged = pipeline._merge_drs_efficiency(base_results, drs).set_index('driver_id')

    assert merged.loc['russell', 'active_aero_efficiency'] == pytest.approx(1.0)
    assert merged.loc['verstappen', 'active_aero_efficiency'] == pytest.approx(0.5)


def test_track_categories_become_binary_columns(pipeline, base_results):
    merged = pipeline._add_track_categories(base_results)

    assert merged['is_power_hungry'].eq(1).all()
    assert merged['is_street_circuit'].eq(0).all()


def test_historical_weights_favour_recent_seasons(pipeline):
    df = pd.DataFrame({'year': [2022, 2023, 2024, 2025]})

    weighted = pipeline._apply_historical_weights(df)

    assert weighted['historical_weight'].is_monotonic_increasing


def test_dataset_round_trips_through_disk(pipeline, base_results, tmp_path):
    path = tmp_path / "training_data.parquet"

    pipeline.save_dataset(base_results, str(path))
    loaded = pipeline.load_dataset(str(path))

    assert len(loaded) == len(base_results)
    assert path.with_suffix('.csv').exists()


def test_empty_source_data_raises_a_clear_error(pipeline, monkeypatch):
    monkeypatch.setattr(pipeline, 'fetch_all_data',
                        lambda years=None, use_fastf1=None: {
                            'ergast': {'race_results': pd.DataFrame()}, 'fastf1': {}})

    with pytest.raises(RuntimeError, match="No race results"):
        pipeline.build_training_dataset([2024])


def test_build_training_dataset_from_stubbed_sources(pipeline, base_results, monkeypatch):
    quali = pd.DataFrame([
        {'year': 2024, 'round': 1, 'driver_id': 'russell', 'quali_position': 1},
        {'year': 2024, 'round': 1, 'driver_id': 'verstappen', 'quali_position': 2},
    ])
    reliability = pd.DataFrame([
        {'constructor_id': 'mercedes', 'reliability_score': 0.97},
        {'constructor_id': 'red_bull', 'reliability_score': 0.80},
    ])

    monkeypatch.setattr(pipeline, 'fetch_all_data', lambda years=None, use_fastf1=None: {
        'ergast': {
            'race_results': base_results.copy(),
            'qualifying': quali,
            'pit_stops': pd.DataFrame(),
            'reliability': reliability,
        },
        'fastf1': {},
    })

    df = pipeline.build_training_dataset([2024], use_fastf1=False)

    assert len(df) == 2
    assert 'position_delta' in df.columns
    assert 'active_aero_efficiency' in df.columns
    assert 'historical_weight' in df.columns
    # A retirement is treated as a P20 finish for the delta.
    assert df.set_index('driver_id').loc['verstappen', 'position_delta'] == 2 - 20


def test_fastf1_is_skipped_when_disabled(pipeline, monkeypatch):
    """The FastF1 import is lazy; disabling it must avoid that code path."""
    def explode(*args, **kwargs):
        raise AssertionError("FastF1 should not be touched when disabled")

    monkeypatch.setattr('src.data.ergast_api.fetch_ergast_data',
                        lambda *a, **k: {'race_results': pd.DataFrame()})
    monkeypatch.setattr('src.data.pipeline.fetch_ergast_data',
                        lambda *a, **k: {'race_results': pd.DataFrame()})
    monkeypatch.setattr('src.data.fastf1_client.fetch_fastf1_data', explode)

    data = pipeline.fetch_all_data([2024], use_fastf1=False)

    assert data['fastf1'] == {}
