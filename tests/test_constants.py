"""Naming and circuit lookups shared across the project."""

import pytest

from src.constants import (
    DEFAULT_RACE_LAPS,
    GRID_2026,
    TEAM_ENGINE_2026,
    constructor_display_name,
    constructor_id_from_name,
    race_laps_for,
)


def test_grid_2026_is_a_full_twenty_car_field():
    assert len(GRID_2026) == 20

    driver_ids = [entry[0] for entry in GRID_2026]
    assert len(set(driver_ids)) == 20

    codes = [entry[1] for entry in GRID_2026]
    assert len(set(codes)) == 20
    assert all(len(code) == 3 and code.isupper() for code in codes)


def test_every_2026_team_fields_exactly_two_cars():
    counts = {}
    for _, _, _, constructor_id in GRID_2026:
        counts[constructor_id] = counts.get(constructor_id, 0) + 1

    assert set(counts) == set(TEAM_ENGINE_2026)
    assert set(counts.values()) == {2}


@pytest.mark.parametrize("constructor_id,expected", [
    ('red_bull', 'Red Bull Racing'),
    ('sauber', 'Audi (Sauber)'),
    ('kick_sauber', 'Audi (Sauber)'),
    ('rb', 'RB'),
    ('Mercedes', 'Mercedes'),
])
def test_constructor_display_name(constructor_id, expected):
    assert constructor_display_name(constructor_id) == expected


def test_constructor_display_name_is_idempotent():
    for constructor_id, _, _, _ in [(c, '', '', '') for c in TEAM_ENGINE_2026]:
        once = constructor_display_name(constructor_id)
        assert constructor_display_name(once) == once


def test_display_names_round_trip_back_to_ids():
    for constructor_id in TEAM_ENGINE_2026:
        display = constructor_display_name(constructor_id)
        assert constructor_id_from_name(display) == constructor_id


def test_constructor_display_name_handles_missing_input():
    assert constructor_display_name('') == 'Unknown'
    assert constructor_display_name(None) == 'Unknown'


@pytest.mark.parametrize("circuit,laps", [
    ('Bahrain International Circuit', 57),
    ('Circuit de Monaco', 78),
    ('Autodromo Nazionale di Monza', 53),
    ('Circuit de Spa-Francorchamps', 44),
])
def test_race_laps_for_known_circuits(circuit, laps):
    assert race_laps_for(circuit) == laps


def test_race_laps_falls_back_for_unknown_circuits():
    assert race_laps_for('Somewhere New') == DEFAULT_RACE_LAPS
    assert race_laps_for(None) == DEFAULT_RACE_LAPS
