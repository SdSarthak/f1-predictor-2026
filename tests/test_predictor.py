"""End-to-end prediction, from a trained model through to formatted output."""

import json

import pandas as pd
import pytest

from src.predictor import F1Predictor


@pytest.fixture
def predictor(engineered, config_path, tmp_path):
    """A trained predictor backed by the synthetic dataset, Elo disabled."""
    df, engineer = engineered

    predictor = F1Predictor(config_path, use_elo=False)
    predictor.feature_engineer = engineer
    predictor.training_data = df
    predictor.train(df, model_type='xgboost', save=False)

    return predictor


@pytest.fixture
def prediction(predictor):
    return predictor.predict_race(
        year=2026, round_num=1, circuit_name='Bahrain',
        run_simulation=True, n_simulations=120,
    )


def test_prediction_features_are_actually_built(predictor):
    """Regression test: the builder used to return an empty DataFrame."""
    features = predictor._build_prediction_features(2026, 1, 'Bahrain', None, None)

    assert not features.empty
    assert len(features) == predictor.training_data['driver_id'].nunique()
    assert features['driver_id'].is_unique


def test_prediction_features_cover_every_model_input(predictor):
    features = predictor._build_prediction_features(2026, 1, 'Bahrain', None, None)

    missing = set(predictor.feature_engineer.get_feature_columns()) - set(features.columns)
    assert not missing
    assert not features[predictor.model_trainer.feature_columns].isna().any().any()


def test_supplied_grid_positions_are_respected(predictor):
    drivers = list(predictor.training_data['driver_id'].unique())
    grid = {driver: i + 1 for i, driver in enumerate(drivers)}

    features = predictor._build_prediction_features(2026, 1, 'Bahrain', None, grid)
    by_driver = features.set_index('driver_id')['grid_position']

    for driver, position in grid.items():
        assert by_driver[driver] == position

    pole = features[features['grid_position'] == 1].iloc[0]
    assert pole['is_front_row'] == 1
    assert pole['is_top5_grid'] == 1


def test_a_partial_grid_is_completed_for_the_rest_of_the_field(predictor):
    """Supplying only the front row must still produce a full, unique grid."""
    features = predictor._build_prediction_features(
        2026, 1, 'Bahrain', None, {'russell': 1, 'norris': 2})

    by_driver = features.set_index('driver_id')['grid_position']
    assert by_driver['russell'] == 1
    assert by_driver['norris'] == 2

    positions = sorted(features['grid_position'])
    assert positions == [float(i) for i in range(1, len(features) + 1)]


def test_partial_grid_is_reported_in_the_result(predictor):
    result = predictor.predict_race(2026, 1, 'Bahrain', grid_positions={'russell': 1},
                                    run_simulation=False)

    assert set(result['grid_positions']) == set(result['adjusted_predictions'])
    assert result['grid_positions']['russell'] == 1


def test_grid_is_estimated_when_qualifying_is_unknown(predictor):
    features = predictor._build_prediction_features(2026, 1, 'Bahrain', None, None)
    positions = sorted(features['grid_position'])

    assert positions == [float(i) for i in range(1, len(features) + 1)]


def test_circuit_flags_follow_the_named_circuit(predictor):
    monaco = predictor._build_prediction_features(2026, 1, 'Circuit de Monaco', None, None)
    monza = predictor._build_prediction_features(2026, 1, 'Monza', None, None)

    assert monaco['is_street_circuit'].eq(1).all()
    assert monaco['is_power_hungry'].eq(0).all()
    assert monza['is_power_hungry'].eq(1).all()


def test_a_specific_driver_subset_can_be_requested(predictor):
    features = predictor._build_prediction_features(2026, 1, 'Bahrain', ['russell', 'norris'], None)

    assert sorted(features['driver_id']) == ['norris', 'russell']


def test_unknown_drivers_still_get_a_row(predictor):
    features = predictor._build_prediction_features(2026, 1, 'Bahrain', ['russell', 'bortoleto'], None)

    assert sorted(features['driver_id']) == ['bortoleto', 'russell']
    assert features['constructor_id'].ne('unknown').all()


def test_field_falls_back_to_the_2026_grid_without_history(config_path):
    predictor = F1Predictor(config_path, use_elo=False)
    entries = predictor._resolve_grid_entries(2026, None)

    assert len(entries) == 20
    assert {'verstappen', 'russell', 'bortoleto'} <= {e['driver_id'] for e in entries}


def test_predicting_before_training_is_rejected(config_path):
    predictor = F1Predictor(config_path, use_elo=False)

    with pytest.raises(ValueError, match="not trained"):
        predictor.predict_race(year=2026, round_num=1)


def test_prediction_result_has_the_expected_shape(predictor, prediction):
    drivers = set(prediction['adjusted_predictions'])

    assert prediction['year'] == 2026
    assert prediction['round'] == 1
    assert prediction['circuit'] == 'Bahrain'
    assert prediction['circuit_type'] == 'power_hungry'
    assert drivers
    for key in ('uncertainties', 'drivers', 'constructors', 'grid_positions', '2026_adjustments'):
        assert set(prediction[key]) == drivers


def test_predicted_positions_are_on_the_grid(prediction):
    for position in prediction['adjusted_predictions'].values():
        assert 1.0 <= position <= 20.0
    for uncertainty in prediction['uncertainties'].values():
        assert uncertainty > 0


def test_monte_carlo_block_is_complete(prediction):
    mc = prediction['monte_carlo']
    drivers = set(prediction['adjusted_predictions'])

    assert set(mc['win_probabilities']) == drivers
    assert sum(mc['win_probabilities'].values()) == pytest.approx(1.0, abs=1e-9)
    assert set(prediction['confidence_intervals']) == drivers


def test_simulation_can_be_skipped(predictor):
    result = predictor.predict_race(year=2026, round_num=1, circuit_name='Bahrain',
                                    run_simulation=False)

    assert 'monte_carlo' not in result
    assert result['adjusted_predictions']


def test_reliability_falls_back_per_team(predictor):
    scores = predictor._get_reliability_scores(
        ['russell', 'verstappen', 'nobody'],
        {'russell': 'mercedes', 'verstappen': 'red_bull'},
    )

    assert set(scores) == {'russell', 'verstappen', 'nobody'}
    assert all(0.5 <= value <= 1.0 for value in scores.values())
    # config/settings.yaml rates Mercedes above Red Bull for 2026.
    assert scores['russell'] > scores['verstappen']


def test_race_distance_is_looked_up_per_circuit(predictor):
    assert predictor._get_race_laps('Circuit de Monaco') == 78
    assert predictor._get_race_laps('Bahrain International Circuit') == 57
    assert predictor._get_race_laps(None) == 57


def test_formatted_output_names_drivers_and_teams(predictor, prediction):
    text = predictor.format_predictions(prediction)

    assert 'Race Prediction: Bahrain' in text
    assert 'Win %' in text
    assert '90% Confidence Intervals' in text

    first_driver = min(prediction['monte_carlo']['expected_positions'],
                       key=prediction['monte_carlo']['expected_positions'].get)
    assert prediction['drivers'][first_driver] in text


def test_formatted_output_without_a_simulation(predictor):
    result = predictor.predict_race(year=2026, round_num=1, circuit_name='Bahrain',
                                    run_simulation=False)

    text = predictor.format_predictions(result)

    assert 'Predicted Order' in text


def test_predictions_are_saved_as_valid_json(predictor, prediction, tmp_path):
    path = predictor.save_predictions(prediction, path=str(tmp_path))

    assert path.exists()
    saved = json.loads(path.read_text(encoding='utf-8'))

    assert saved['circuit'] == 'Bahrain'
    assert set(saved['adjusted_predictions']) == set(prediction['adjusted_predictions'])


def test_save_predictions_sanitises_the_circuit_name(predictor, prediction, tmp_path):
    prediction = dict(prediction, circuit='Sao Paulo / Interlagos')

    path = predictor.save_predictions(prediction, path=str(tmp_path))

    assert path.parent == tmp_path
    assert '/' not in path.name


def test_elo_blending_changes_the_prediction(engineered, config_path, tmp_path):
    df, engineer = engineered

    ratings_path = tmp_path / "ratings.json"
    plain = F1Predictor(config_path, use_elo=False)
    plain.feature_engineer = engineer
    plain.training_data = df
    plain.train(df, model_type='xgboost', save=False)

    blended = F1Predictor(config_path, use_elo=True)
    blended.race_updater.ratings_path = str(ratings_path)
    blended.feature_engineer = engineer
    blended.training_data = df
    blended.model_trainer = plain.model_trainer
    blended.is_trained = True
    blended.race_updater.update_from_race(
        pd.DataFrame([
            {'driver_id': 'magnussen', 'constructor_id': 'haas', 'position': 1, 'dnf': False},
            {'driver_id': 'verstappen', 'constructor_id': 'red_bull', 'position': 20, 'dnf': True},
        ]),
        circuit='Bahrain', race_number=1,
    )

    without = plain.predict_race(2026, 1, 'Bahrain', run_simulation=False)
    with_elo = blended.predict_race(2026, 1, 'Bahrain', run_simulation=False)

    assert with_elo['elo_weight'] > 0
    assert with_elo['adjusted_predictions'] != without['adjusted_predictions']


def test_update_after_race_requires_the_core_columns(predictor):
    with pytest.raises(ValueError, match="missing required columns"):
        predictor.update_after_race(pd.DataFrame({'driver_id': ['russell']}),
                                    circuit='Bahrain', round_num=1)
