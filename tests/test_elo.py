"""Elo rating system and the ML/Elo blending layer."""

import json

import pandas as pd
import pytest

from src.models.elo_updater import F1EloRatingSystem
from src.models.race_updater import RaceByRaceUpdater


def race_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {'driver_id': 'russell', 'constructor_id': 'mercedes', 'position': 1,
         'quali_position': 1, 'dnf': False},
        {'driver_id': 'norris', 'constructor_id': 'mclaren', 'position': 2,
         'quali_position': 2, 'dnf': False},
        {'driver_id': 'leclerc', 'constructor_id': 'ferrari', 'position': 3,
         'quali_position': 3, 'dnf': False},
        {'driver_id': 'verstappen', 'constructor_id': 'red_bull', 'position': 4,
         'quali_position': 4, 'dnf': True},
    ])


def test_expected_score_is_a_half_for_equal_ratings():
    elo = F1EloRatingSystem()
    assert elo.expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_scores_are_complementary():
    elo = F1EloRatingSystem()
    assert elo.expected_score(1700, 1500) + elo.expected_score(1500, 1700) == pytest.approx(1.0)


def test_a_four_hundred_point_edge_is_ten_to_one():
    elo = F1EloRatingSystem()
    assert elo.expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-6)


def test_unknown_entities_start_at_the_initial_rating():
    elo = F1EloRatingSystem(initial_driver_rating=1500, initial_constructor_rating=1400)

    assert elo.get_driver_rating('nobody') == 1500
    assert elo.get_constructor_rating('nobody') == 1400


def test_combined_rating_respects_the_weighting():
    elo = F1EloRatingSystem()
    elo.driver_ratings['a'] = 1600
    elo.constructor_ratings['x'] = 1400

    combined = elo.get_combined_rating('a', 'x', driver_weight=0.3, constructor_weight=0.7)

    assert combined == pytest.approx(1600 * 0.3 + 1400 * 0.7)


def test_a_race_result_separates_the_winner_from_the_loser():
    elo = F1EloRatingSystem()
    elo.update_from_race_result(race_frame(), circuit='Bahrain', race_number=1)

    assert elo.get_constructor_rating('mercedes') > elo.get_constructor_rating('red_bull')
    assert elo.get_driver_rating('russell') > elo.get_driver_rating('verstappen')
    assert elo.race_counter == 1
    assert len(elo.rating_history) == 1


def test_win_probabilities_sum_to_one():
    elo = F1EloRatingSystem()
    elo.update_from_race_result(race_frame(), circuit='Bahrain', race_number=1)

    pairs = [('russell', 'mercedes'), ('norris', 'mclaren'), ('verstappen', 'red_bull')]
    probabilities = elo.get_win_probabilities(pairs)

    assert set(probabilities) == {'russell', 'norris', 'verstappen'}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities['russell'] > probabilities['verstappen']


def test_season_decay_pulls_ratings_toward_the_baseline():
    elo = F1EloRatingSystem(decay_factor=0.5)
    elo.driver_ratings['a'] = 1900
    elo.constructor_ratings['x'] = 1100

    elo.apply_season_decay()

    assert elo.driver_ratings['a'] == pytest.approx(1700)
    assert elo.constructor_ratings['x'] == pytest.approx(1300)


def test_practice_update_favours_the_faster_car():
    elo = F1EloRatingSystem()
    practice = pd.DataFrame([
        {'driver_id': 'a', 'constructor_id': 'fast', 'best_lap_time': 90.0},
        {'driver_id': 'b', 'constructor_id': 'slow', 'best_lap_time': 91.0},
    ])

    elo.update_from_practice_quali(practice, circuit='Bahrain')

    adjustments = elo.circuit_adjustments['Bahrain']
    assert adjustments['fast'] > adjustments['slow']


def test_ratings_survive_a_save_load_round_trip(tmp_path):
    elo = F1EloRatingSystem()
    elo.update_from_race_result(race_frame(), circuit='Bahrain', race_number=1)

    path = tmp_path / "ratings.json"
    elo.save_ratings(str(path))

    saved = json.loads(path.read_text())
    assert 'driver_ratings' in saved and 'constructor_ratings' in saved

    reloaded = F1EloRatingSystem()
    reloaded.load_ratings(str(path))

    assert reloaded.driver_ratings == elo.driver_ratings
    assert reloaded.constructor_ratings == elo.constructor_ratings
    assert reloaded.race_counter == elo.race_counter


def test_rating_summary_lists_drivers_and_constructors():
    elo = F1EloRatingSystem()
    elo.update_from_race_result(race_frame(), circuit='Bahrain', race_number=1)

    summary = elo.get_rating_summary()

    assert set(summary['type']) == {'driver', 'constructor'}
    assert summary['rating'].is_monotonic_decreasing


def test_updater_starts_clean_when_no_ratings_file_exists(tmp_path):
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "missing.json"))

    assert updater.has_ratings is False


def test_blending_is_a_no_op_without_rating_history(tmp_path):
    """Without history every Elo rating is identical, so the order is arbitrary."""
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "missing.json"))

    ml = {'russell': 2.0, 'verstappen': 3.0, 'norris': 4.0}
    uncertainties = {driver: 2.0 for driver in ml}
    teams = {'russell': 'mercedes', 'verstappen': 'red_bull', 'norris': 'mclaren'}

    blended, blended_unc = updater.blend_predictions(ml, uncertainties, teams)

    assert blended == ml
    assert blended_unc == uncertainties


def test_blending_pulls_predictions_toward_recent_form(tmp_path):
    path = tmp_path / "ratings.json"
    updater = RaceByRaceUpdater(elo_weight=0.5, ratings_path=str(path))
    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    ml = {'russell': 5.0, 'verstappen': 1.0}
    uncertainties = {'russell': 2.0, 'verstappen': 2.0}
    teams = {'russell': 'mercedes', 'verstappen': 'red_bull'}

    blended, _ = updater.blend_predictions(ml, uncertainties, teams)

    # Elo saw Russell win and Verstappen retire, so the gap must narrow.
    assert blended['russell'] < ml['russell']
    assert blended['verstappen'] > ml['verstappen']


def test_blending_widens_uncertainty_when_the_two_views_disagree(tmp_path):
    path = tmp_path / "ratings.json"
    updater = RaceByRaceUpdater(elo_weight=0.5, ratings_path=str(path))
    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    ml = {'russell': 1.0, 'verstappen': 2.0, 'norris': 3.0, 'leclerc': 4.0}
    uncertainties = {driver: 2.0 for driver in ml}
    teams = {'russell': 'mercedes', 'verstappen': 'red_bull',
             'norris': 'mclaren', 'leclerc': 'ferrari'}

    _, blended_unc = updater.blend_predictions(ml, uncertainties, teams)

    # Verstappen: ML says P2, Elo says last -> the disagreement must show up.
    assert blended_unc['verstappen'] > uncertainties['verstappen']
    assert blended_unc['russell'] < uncertainties['russell']


def test_constructor_ids_and_display_names_update_the_same_rating(tmp_path):
    """`red_bull` and `Red Bull Racing` must not become two separate entities."""
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "ratings.json"))
    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    ratings = updater.elo_system.constructor_ratings

    assert 'Red Bull Racing' in ratings
    assert 'red_bull' not in ratings


def test_updater_persists_ratings_between_instances(tmp_path):
    path = tmp_path / "ratings.json"

    first = RaceByRaceUpdater(ratings_path=str(path))
    first.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    second = RaceByRaceUpdater(ratings_path=str(path))

    assert second.has_ratings
    assert second.elo_system.constructor_ratings == first.elo_system.constructor_ratings


def test_elo_weight_grows_as_the_season_progresses(tmp_path):
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "ratings.json"))

    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)
    early = updater.elo_weight

    updater.update_from_race(race_frame(), circuit='Monza', race_number=12)
    late = updater.elo_weight

    assert late > early
    assert updater.ml_weight == pytest.approx(1.0 - late)


def test_power_rankings_are_normalised(tmp_path):
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "ratings.json"))
    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    rankings = updater.get_constructor_power_rankings()

    assert min(rankings.values()) == pytest.approx(0.0)
    assert max(rankings.values()) == pytest.approx(1.0)


def test_monte_carlo_parameter_adjustments_stay_in_range(tmp_path):
    updater = RaceByRaceUpdater(ratings_path=str(tmp_path / "ratings.json"))
    updater.update_from_race(race_frame(), circuit='Bahrain', race_number=1)

    params = updater.adjust_monte_carlo_params('russell', 'mercedes')

    assert 0.3 <= params['overtake_skill'] <= 0.7
    assert 0.01 <= params['mistake_probability'] <= 0.03
    assert 0.85 <= params['reliability_mult'] <= 1.15
