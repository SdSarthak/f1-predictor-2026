"""2026 regulation adjustments."""

import pandas as pd
import pytest

from src.constants import TEAM_ENGINE_2026
from src.regulations.rules_2026 import Regulations2026, apply_2026_adjustments

TEAMS = list(TEAM_ENGINE_2026)


@pytest.fixture
def regs(config_path):
    return Regulations2026(config_path)


def test_grid_entries_cover_the_whole_field(regs):
    entries = regs.get_2026_grid_entries()

    assert len(entries) == 20
    for entry in entries:
        assert entry['driver_id']
        assert entry['constructor_id'] in TEAM_ENGINE_2026
        assert entry['constructor_name'] != 'Unknown'


def test_new_engine_manufacturers_carry_extra_early_season_uncertainty(regs):
    audi = regs.get_new_engine_uncertainty('sauber', race_number=1)
    mercedes = regs.get_new_engine_uncertainty('mercedes', race_number=1)

    assert audi > 1.0
    assert mercedes == 1.0


def test_new_engine_uncertainty_decays_over_the_season(regs):
    early = regs.get_new_engine_uncertainty('sauber', race_number=1)
    later = regs.get_new_engine_uncertainty('sauber', race_number=4)
    settled = regs.get_new_engine_uncertainty('sauber', race_number=20)

    assert early > later >= 1.0
    assert settled == 1.0


@pytest.mark.parametrize("team", TEAMS)
def test_team_uncertainty_components_are_positive(regs, team):
    uncertainty = regs.calculate_2026_team_uncertainty(team, race_number=1)

    assert set(uncertainty) == {'total', 'engine', 'driver', 'season', 'testing'}
    assert all(value > 0 for value in uncertainty.values())


def test_season_uncertainty_falls_as_the_season_settles(regs):
    opener = regs.calculate_2026_team_uncertainty('mercedes', race_number=1)
    midseason = regs.calculate_2026_team_uncertainty('mercedes', race_number=12)

    assert opener['season'] > midseason['season']
    assert opener['total'] > midseason['total']


@pytest.mark.parametrize("team", TEAMS)
def test_power_unit_profile_is_bounded(regs, team):
    profile = regs.calculate_power_unit_2026_profile(team, 'power_hungry')

    assert 0.0 < profile['battery_score'] <= 1.0
    assert 0.0 < profile['ice_score'] <= 1.0
    assert min(profile['battery_score'], profile['ice_score']) <= profile['combined_pu_score']
    assert profile['combined_pu_score'] <= max(profile['battery_score'], profile['ice_score'])


def test_circuit_type_shifts_the_power_unit_weighting(regs):
    power = regs.calculate_power_unit_2026_profile('mercedes', 'power_hungry')
    downforce = regs.calculate_power_unit_2026_profile('mercedes', 'high_downforce')

    assert power['circuit_type'] == 'power_hungry'
    assert power['combined_pu_score'] != downforce['combined_pu_score']


def test_active_aero_efficiency_stays_within_bounds(regs):
    for team in ('Mercedes', 'Haas', 'A Team That Does Not Exist'):
        assert 0.5 <= regs.calculate_active_aero_efficiency(team) <= 1.0


def test_active_aero_ordering_matches_the_config(regs):
    assert (regs.calculate_active_aero_efficiency('Mercedes')
            > regs.calculate_active_aero_efficiency('Aston Martin'))


def test_adjust_predictions_preserves_the_field(regs):
    predictions = {'russell': 2.0, 'verstappen': 3.0, 'bortoleto': 15.0}
    uncertainties = {driver: 1.5 for driver in predictions}
    teams = {'russell': 'mercedes', 'verstappen': 'red_bull', 'bortoleto': 'sauber'}

    adjusted = regs.adjust_predictions_for_2026(predictions, uncertainties, teams, race_number=1)

    assert set(adjusted['predictions']) == set(predictions)
    assert set(adjusted['uncertainties']) == set(predictions)
    assert set(adjusted['adjustments']) == set(predictions)


def test_adjusted_predictions_stay_on_the_grid(regs):
    predictions = {'a': 1.0, 'b': 20.0}
    uncertainties = {'a': 1.0, 'b': 1.0}
    teams = {'a': 'mercedes', 'b': 'sauber'}

    adjusted = regs.adjust_predictions_for_2026(predictions, uncertainties, teams)

    for value in adjusted['predictions'].values():
        assert 1 <= value <= 20


def test_new_engine_teams_end_up_less_certain(regs):
    predictions = {'a': 5.0, 'b': 5.0}
    uncertainties = {'a': 1.5, 'b': 1.5}
    teams = {'a': 'mercedes', 'b': 'sauber'}

    adjusted = regs.adjust_predictions_for_2026(predictions, uncertainties, teams, race_number=1)

    assert adjusted['uncertainties']['b'] > adjusted['uncertainties']['a']


def test_historical_weight_decay_is_applied(regs):
    df = pd.DataFrame({
        'year': [2022, 2023, 2024, 2025],
        'constructor_id': ['mercedes'] * 4,
        'constructor_form': [10.0] * 4,
        'points': [100, 200, 300, 400],
    })

    out = regs.apply_historical_weight_decay(df)

    assert 'reg_weight' in out.columns
    assert out.loc[out['year'] == 2025, 'reg_weight'].iloc[0] > \
           out.loc[out['year'] == 2022, 'reg_weight'].iloc[0]
    assert 'constructor_form_weighted' in out.columns


def test_apply_2026_adjustments_adds_both_columns(config_path):
    df = pd.DataFrame({
        'year': [2021, 2022, 2022, 2021],
        'constructor_id': ['mercedes', 'mercedes', 'ferrari', 'ferrari'],
        'constructor_form': [10.0, 10.0, 8.0, 8.0],
        'points': [400, 300, 100, 200],
    })

    out = apply_2026_adjustments(df, config_path)

    assert 'reg_weight' in out.columns
    assert 'reg_adaptation_score' in out.columns
    assert out['reg_adaptation_score'].between(0, 1).all()


def test_driver_lineup_matches_the_grid_entries(regs):
    lineup = regs.get_2026_driver_lineup()
    entries = regs.get_2026_grid_entries()

    for entry in entries:
        team_drivers = lineup[entry['constructor_name']]
        assert entry['driver_name'] in team_drivers
