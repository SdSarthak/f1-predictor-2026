"""Command-line argument handling."""

import pytest

import run_predictor


def test_parse_grid_builds_a_mapping():
    grid = run_predictor.parse_grid(['russell=1', 'norris=2', 'verstappen=10'])

    assert grid == {'russell': 1, 'norris': 2, 'verstappen': 10}


def test_parse_grid_returns_none_when_unset():
    assert run_predictor.parse_grid(None) is None
    assert run_predictor.parse_grid([]) is None


def test_parse_grid_rejects_entries_without_a_position():
    with pytest.raises(ValueError, match="Use driver_id=position"):
        run_predictor.parse_grid(['russell'])


def test_parse_grid_rejects_non_integer_positions():
    with pytest.raises(ValueError, match="must be an integer"):
        run_predictor.parse_grid(['russell=pole'])


def test_help_output_is_produced_without_arguments(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['run_predictor.py'])

    assert run_predictor.main() == 0

    output = capsys.readouterr().out
    assert 'Quick Start' in output
    assert '--predict' in output
