"""Feature engineering: coverage, leak-freedom and encoder behaviour."""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_engineering import FeatureEngineer, engineer_features


def test_every_declared_feature_column_is_produced(engineered):
    df, engineer = engineered

    missing = set(engineer.get_feature_columns()) - set(df.columns)
    assert not missing, f"declared but never produced: {sorted(missing)}"


def test_feature_matrix_has_no_nans_or_infinities(engineered):
    df, engineer = engineered

    features = df[engineer.get_feature_columns()]
    assert not features.isna().any().any()
    assert np.isfinite(features.to_numpy(dtype=float)).all()


def test_row_count_is_preserved(raw_results, engineered):
    df, _ = engineered
    assert len(df) == len(raw_results)


def test_driver_form_does_not_use_the_current_race():
    """A driver's first ever race must fall back to the neutral prior."""
    engineer = FeatureEngineer({})
    df = _minimal_frame()

    out = engineer.engineer_features(df)
    first_race = out[(out['driver_id'] == 'a') & (out['round'] == 1)].iloc[0]

    assert first_race['driver_form'] == 0.5
    assert first_race['grid_conversion_rate'] == 0.5


def test_grid_conversion_rate_reflects_only_prior_races():
    """
    Driver 'a' gains a place in round 1 and loses one in round 2, so by round 3
    the leak-free conversion rate must be 0.5 - not the 1/3 you would get by
    including round 3 itself.
    """
    engineer = FeatureEngineer({})
    df = _minimal_frame()

    out = engineer.engineer_features(df).sort_values(['driver_id', 'round'])
    a_rows = out[out['driver_id'] == 'a']

    assert a_rows.iloc[0]['grid_conversion_rate'] == 0.5   # no history
    assert a_rows.iloc[1]['grid_conversion_rate'] == 1.0   # gained in R1
    assert a_rows.iloc[2]['grid_conversion_rate'] == 0.5   # gained R1, lost R2


def test_head_to_head_defaults_when_no_teammate_pairs_exist():
    """A single-car field has no teammate battles; the feature must still exist."""
    engineer = FeatureEngineer({})
    df = _minimal_frame()
    df = df[df['driver_id'] == 'a'].copy()

    out = engineer.engineer_features(df)

    assert 'teammate_battle_rate' in out.columns
    assert (out['teammate_battle_rate'] == 0.5).all()


def test_rain_skill_comes_from_config():
    engineer = FeatureEngineer({'rain_performance': {'A': 0.99, 'default': 0.5}})
    df = _minimal_frame()

    out = engineer.engineer_features(df)

    assert out[out['driver_code'] == 'A']['rain_skill'].eq(0.99).all()
    assert out[out['driver_code'] == 'B']['rain_skill'].eq(0.5).all()


def test_transform_categoricals_maps_unseen_labels_to_minus_one(engineered):
    df, engineer = engineered

    known_driver = df['driver_id'].iloc[0]
    new_rows = pd.DataFrame([
        {'driver_id': known_driver, 'constructor_id': 'mercedes', 'circuit_id': 'monza'},
        {'driver_id': 'a_2026_rookie', 'constructor_id': 'audi', 'circuit_id': 'madrid'},
    ])

    encoded = engineer.transform_categoricals(new_rows)

    assert encoded.loc[0, 'driver_id_encoded'] >= 0
    assert encoded.loc[1, 'driver_id_encoded'] == -1
    assert encoded.loc[1, 'circuit_id_encoded'] == -1


def test_transform_categoricals_agrees_with_fitted_encoding(engineered):
    df, engineer = engineered

    sample = df[['driver_id', 'constructor_id', 'circuit_id', 'engine_manufacturer']].head(15)
    re_encoded = engineer.transform_categoricals(sample)

    pd.testing.assert_series_equal(
        re_encoded['driver_id_encoded'],
        df['driver_id_encoded'].head(15).astype(int),
        check_names=False,
    )


def test_engineer_features_helper_returns_frame_and_engineer(raw_results):
    df, engineer = engineer_features(raw_results.copy(), {})

    assert isinstance(df, pd.DataFrame)
    assert isinstance(engineer, FeatureEngineer)
    assert engineer.get_target_column() == 'finish_position'


def _minimal_frame() -> pd.DataFrame:
    """Two drivers on one team across three rounds at a single circuit."""
    records = [
        # round, driver, grid, finish
        (1, 'a', 3, 2),
        (1, 'b', 4, 5),
        (2, 'a', 2, 4),
        (2, 'b', 5, 3),
        (3, 'a', 1, 1),
        (3, 'b', 6, 6),
    ]

    rows = []
    for round_num, driver, grid, finish in records:
        rows.append({
            'year': 2025,
            'round': round_num,
            'driver_id': driver,
            'driver_code': driver.upper(),
            'constructor_id': 'team_x',
            'constructor_name': 'Team X',
            'circuit_id': 'bahrain',
            'circuit_name': 'Bahrain International Circuit',
            'race_name': 'Bahrain Grand Prix',
            'grid_position': grid,
            'finish_position': finish,
            'points': max(0, 11 - finish),
            'reliability_score': 0.9,
        })

    df = pd.DataFrame(rows)
    df['position_delta'] = df['grid_position'] - df['finish_position']
    return df


def test_engineering_is_idempotent_over_an_already_engineered_frame(engineered):
    """
    `save_dataset` persists the engineered frame, so `train()` and prediction
    both re-run the engineer over its own output. The merge-based features used
    to collide into `<name>_x` / `<name>_y`, which raised
    `KeyError: 'overtake_difficulty'` and silently reset `constructor_form`.
    """
    df, engineer = engineered

    again, engineer2 = engineer_features(df.copy(), engineer.config)

    assert not [c for c in again.columns if c.endswith(('_x', '_y'))]
    assert len(again) == len(df)
    for column in engineer.get_feature_columns():
        np.testing.assert_allclose(
            again[column].astype(float).to_numpy(),
            df[column].astype(float).to_numpy(),
            err_msg=f"{column} changed on the second pass",
        )


def test_empty_and_malformed_input_is_rejected_with_a_useful_message():
    engineer = FeatureEngineer({})

    with pytest.raises(ValueError, match="empty dataset"):
        engineer.engineer_features(pd.DataFrame())

    with pytest.raises(ValueError, match="missing required columns"):
        engineer.engineer_features(pd.DataFrame({'driver_id': ['a'], 'round': [1]}))


def test_pit_consistency_survives_a_single_constructor():
    """
    One team means a zero spread in pit times; the old code divided by that
    maximum and produced NaN/inf for every row.
    """
    engineer = FeatureEngineer({})
    df = _minimal_frame()
    df['avg_pit_time'] = 22.5

    out = engineer.engineer_features(df)

    assert np.isfinite(out['pit_consistency'].to_numpy(dtype=float)).all()


def test_missing_circuit_names_do_not_crash_the_power_track_flag():
    """`NaN.lower()` used to raise AttributeError mid-build."""
    engineer = FeatureEngineer({'track_categories': {'power_hungry': ['Monza']}})
    df = _minimal_frame()
    df.loc[0, 'circuit_name'] = np.nan

    out = engineer.engineer_features(df)

    assert set(out['is_power_track'].unique()) <= {0, 1}
