"""Ergast/Jolpica client - parsing and pagination, with no network access."""

import pandas as pd
import pytest

from src.data.ergast_api import ErgastAPI


def build_result(driver: str, constructor: str, position: int) -> dict:
    return {
        'position': str(position),
        'grid': str(position),
        'points': str(max(0, 11 - position)),
        'status': 'Finished',
        'laps': '57',
        'Driver': {
            'driverId': driver,
            'code': driver[:3].upper(),
            'givenName': driver.title(),
            'familyName': 'Racer',
        },
        'Constructor': {'constructorId': constructor, 'name': constructor.title()},
    }


def build_race(season: str, round_num: str, results: list) -> dict:
    return {
        'season': season,
        'round': round_num,
        'raceName': f"Round {round_num} Grand Prix",
        'date': f"{season}-03-01",
        'Circuit': {
            'circuitId': 'bahrain',
            'circuitName': 'Bahrain International Circuit',
            'Location': {'country': 'Bahrain', 'locality': 'Sakhir'},
        },
        'Results': results,
    }


def paged_api(pages, total, monkeypatch) -> ErgastAPI:
    """An ErgastAPI whose `_make_request` serves canned pages."""
    api = ErgastAPI(cache_enabled=False, request_delay=0)
    calls = []

    def fake_request(endpoint, limit=None, offset=0):
        calls.append(offset)
        index = offset // ErgastAPI.PAGE_SIZE
        races = pages[index] if index < len(pages) else []
        return {'MRData': {'total': str(total), 'limit': str(limit), 'offset': str(offset),
                           'RaceTable': {'Races': races}}}

    monkeypatch.setattr(api, '_make_request', fake_request)
    api.calls = calls
    return api


def test_single_page_results_are_parsed(monkeypatch):
    race = build_race('2024', '1', [build_result('russell', 'mercedes', 1),
                                    build_result('norris', 'mclaren', 2)])
    api = paged_api([[race]], total=2, monkeypatch=monkeypatch)

    df = api.get_race_results(2024)

    assert len(df) == 2
    assert list(df['driver_id']) == ['russell', 'norris']
    assert df['year'].eq(2024).all()
    assert df['circuit_id'].eq('bahrain').all()
    assert df.loc[0, 'grid_position'] == 1


def test_results_are_paged_until_total_is_reached(monkeypatch):
    """The mirror caps `limit` at 100, so a full season needs several pages."""
    pages = []
    for page_index in range(3):
        races = []
        for offset in range(5):
            round_num = page_index * 5 + offset + 1
            races.append(build_race('2024', str(round_num),
                                    [build_result(f'driver_{i}', 'team', i + 1)
                                     for i in range(20)]))
        pages.append(races)

    api = paged_api(pages, total=300, monkeypatch=monkeypatch)
    df = api.get_race_results(2024)

    assert api.calls == [0, 100, 200]
    assert len(df) == 300
    assert df['round'].nunique() == 15


def test_a_race_split_across_a_page_boundary_is_stitched_back_together(monkeypatch):
    """Round 5's results straddle two pages and must not become two races."""
    first_page = [build_race('2024', '5', [build_result(f'd{i}', 'team', i + 1)
                                           for i in range(12)])]
    second_page = [build_race('2024', '5', [build_result(f'd{i}', 'team', i + 1)
                                            for i in range(12, 20)]),
                   build_race('2024', '6', [build_result('d0', 'team', 1)])]

    api = paged_api([first_page, second_page], total=200, monkeypatch=monkeypatch)
    df = api.get_race_results(2024)

    round_five = df[df['round'] == 5]
    assert len(round_five) == 20
    assert round_five['driver_id'].nunique() == 20
    assert len(df) == 21


def test_pagination_stops_once_the_total_is_covered(monkeypatch):
    race = build_race('2024', '1', [build_result('russell', 'mercedes', 1)])
    api = paged_api([[race]], total=1, monkeypatch=monkeypatch)

    api.get_race_results(2024)

    assert api.calls == [0]


def test_non_numeric_finishing_positions_become_null(monkeypatch):
    result = build_result('russell', 'mercedes', 1)
    result['position'] = 'R'
    result['status'] = 'Engine'
    api = paged_api([[build_race('2024', '1', [result])]], total=1, monkeypatch=monkeypatch)

    df = api.get_race_results(2024)

    assert pd.isna(df.loc[0, 'finish_position'])
    assert df.loc[0, 'status'] == 'Engine'


def test_reliability_scores_come_from_dnf_rates(monkeypatch):
    finished = build_result('russell', 'mercedes', 1)
    retired = build_result('verstappen', 'red_bull', 2)
    retired['status'] = 'Engine'

    api = paged_api([[build_race('2024', '1', [finished, retired])]],
                    total=2, monkeypatch=monkeypatch)

    reliability = api.calculate_reliability_scores([2024]).set_index('constructor_id')

    assert reliability.loc['mercedes', 'reliability_score'] == 1.0
    assert reliability.loc['red_bull', 'reliability_score'] == 0.0
    assert reliability.loc['red_bull', 'dnf_rate'] == 1.0


@pytest.mark.parametrize("raw,seconds", [
    ('23.512', 23.512),
    ('1:02.400', 62.4),
    ('nonsense', None),
])
def test_pit_duration_parsing(raw, seconds):
    api = ErgastAPI(cache_enabled=False, request_delay=0)
    parsed = api._parse_pit_duration(raw)

    if seconds is None:
        assert pd.isna(parsed)
    else:
        assert parsed == pytest.approx(seconds)


def test_base_url_can_be_overridden_by_argument_or_environment(monkeypatch):
    assert ErgastAPI(base_url='https://example.test/f1/').base_url == 'https://example.test/f1'

    monkeypatch.setenv('F1_ERGAST_BASE_URL', 'https://from-env.test/f1')
    assert ErgastAPI().base_url == 'https://from-env.test/f1'


def test_empty_race_table_yields_an_empty_frame(monkeypatch):
    api = paged_api([[]], total=0, monkeypatch=monkeypatch)

    assert api.get_race_results(2024).empty
