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


def test_parse_grid_rejects_duplicate_positions():
    with pytest.raises(ValueError, match="claimed by both"):
        run_predictor.parse_grid(['russell=1', 'norris=1'])


def test_parse_grid_rejects_out_of_range_positions():
    with pytest.raises(ValueError, match="between 1 and"):
        run_predictor.parse_grid(['russell=0'])
    with pytest.raises(ValueError, match="between 1 and"):
        run_predictor.parse_grid(['russell=-3'])
    with pytest.raises(ValueError, match="between 1 and"):
        run_predictor.parse_grid([f'russell={run_predictor.MAX_GRID_SLOTS + 1}'])


def test_parse_grid_rejects_a_repeated_driver():
    with pytest.raises(ValueError, match="appears twice"):
        run_predictor.parse_grid(['russell=1', 'russell=2'])


def test_parse_grid_rejects_a_blank_driver_id():
    with pytest.raises(ValueError, match="no driver id"):
        run_predictor.parse_grid(['=1'])


def test_predict_rejects_nonsense_run_parameters():
    with pytest.raises(ValueError, match="--simulations"):
        run_predictor.predict_race(simulations=0)
    with pytest.raises(ValueError, match="--round"):
        run_predictor.predict_race(round_num=0)
    with pytest.raises(ValueError, match="--year"):
        run_predictor.predict_race(year=1899)


def test_train_rejects_implausible_seasons():
    with pytest.raises(ValueError, match="implausible seasons"):
        run_predictor.train_model(years=[2024, 3000])


def test_update_from_results_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        run_predictor.update_from_results(str(tmp_path / "nope.csv"), 'Bahrain', 1)


def test_update_from_results_rejects_an_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("driver_id,constructor_id,position\n", encoding='utf-8')

    with pytest.raises(ValueError, match="no rows"):
        run_predictor.update_from_results(str(path), 'Bahrain', 1)
