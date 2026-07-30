"""Monte Carlo race simulation."""

import numpy as np
import pytest

from src.simulation.monte_carlo import MonteCarloSimulator

DRIVERS = [f"driver_{i:02d}" for i in range(20)]


@pytest.fixture
def simulator(config_path):
    return MonteCarloSimulator(config_path)


@pytest.fixture
def race_inputs():
    predicted = {driver: float(i + 1) for i, driver in enumerate(DRIVERS)}
    uncertainty = {driver: 1.5 for driver in DRIVERS}
    grid = {driver: i + 1 for i, driver in enumerate(DRIVERS)}
    reliability = {driver: 0.92 for driver in DRIVERS}
    return predicted, uncertainty, grid, reliability


@pytest.fixture
def mc_results(simulator, race_inputs):
    np.random.seed(7)
    predicted, uncertainty, grid, reliability = race_inputs
    return simulator.run_monte_carlo(
        predicted, uncertainty, grid, reliability,
        race_laps=57, circuit_type='normal', n_simulations=300,
    )


def test_single_race_assigns_every_driver_a_unique_position(simulator, race_inputs):
    np.random.seed(1)
    predicted, uncertainty, grid, reliability = race_inputs

    result = simulator.simulate_race(predicted, uncertainty, grid, reliability)
    final = result['final_positions']

    assert set(final) == set(DRIVERS)
    assert sorted(final.values()) == list(range(1, len(DRIVERS) + 1))


def test_dnf_drivers_are_classified_behind_every_finisher(simulator, race_inputs):
    np.random.seed(3)
    predicted, uncertainty, grid, reliability = race_inputs
    reliability = {driver: 0.2 for driver in DRIVERS}  # force retirements

    result = simulator.simulate_race(predicted, uncertainty, grid, reliability)
    final = result['final_positions']
    dnfs = set(result['dnf_drivers'])

    assert dnfs, "very low reliability should produce at least one retirement"

    finisher_positions = [p for d, p in final.items() if d not in dnfs]
    dnf_positions = [p for d, p in final.items() if d in dnfs]
    if finisher_positions:
        assert min(dnf_positions) > max(finisher_positions)


def test_win_probabilities_form_a_distribution(mc_results):
    total = sum(mc_results['win_probabilities'].values())
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 <= p <= 1.0 for p in mc_results['win_probabilities'].values())


def test_podium_probabilities_sum_to_three_races_worth(mc_results):
    assert sum(mc_results['podium_probabilities'].values()) == pytest.approx(3.0, abs=1e-9)
    assert sum(mc_results['points_probabilities'].values()) == pytest.approx(10.0, abs=1e-9)


def test_probability_ordering_follows_predicted_strength(mc_results):
    """The predicted front-runner should out-score the predicted backmarker."""
    assert mc_results['win_probabilities'][DRIVERS[0]] > mc_results['win_probabilities'][DRIVERS[-1]]
    assert mc_results['expected_positions'][DRIVERS[0]] < mc_results['expected_positions'][DRIVERS[-1]]


def test_every_driver_appears_in_every_output_block(mc_results):
    for key in ('win_probabilities', 'podium_probabilities', 'points_probabilities',
                'expected_positions', 'position_std', 'dnf_probability',
                'position_distributions'):
        assert set(mc_results[key]) == set(DRIVERS), key


def test_expected_positions_stay_inside_the_field(mc_results):
    for position in mc_results['expected_positions'].values():
        assert 1.0 <= position <= len(DRIVERS)


def test_confidence_intervals_are_ordered_and_in_range(simulator, mc_results):
    intervals = simulator.get_confidence_intervals(mc_results['position_distributions'])

    assert set(intervals) == set(DRIVERS)
    for low, high in intervals.values():
        assert 1 <= low <= high <= len(DRIVERS)


def test_wider_confidence_setting_gives_wider_intervals(simulator, mc_results):
    narrow = simulator.get_confidence_intervals(mc_results['position_distributions'], confidence=0.5)
    wide = simulator.get_confidence_intervals(mc_results['position_distributions'], confidence=0.99)

    narrow_width = sum(high - low for low, high in narrow.values())
    wide_width = sum(high - low for low, high in wide.values())
    assert wide_width >= narrow_width


def test_street_circuits_get_more_safety_cars(simulator):
    assert simulator._get_safety_car_prob('street') > simulator._get_safety_car_prob('normal')
    assert simulator._get_safety_car_prob('normal') > simulator._get_safety_car_prob('high_speed')


def test_lower_reliability_raises_the_dnf_rate(simulator, race_inputs):
    np.random.seed(11)
    predicted, uncertainty, grid, _ = race_inputs

    reliable = simulator.run_monte_carlo(
        predicted, uncertainty, grid, {d: 0.99 for d in DRIVERS}, n_simulations=150)
    fragile = simulator.run_monte_carlo(
        predicted, uncertainty, grid, {d: 0.60 for d in DRIVERS}, n_simulations=150)

    assert (sum(fragile['dnf_probability'].values())
            > sum(reliable['dnf_probability'].values()))


def test_championship_simulation_returns_a_distribution(simulator, mc_results):
    outcome = simulator.simulate_championship([mc_results, mc_results], n_simulations=100)

    probabilities = outcome['championship_probabilities']
    assert probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert outcome['sorted_probabilities'][0][1] == max(probabilities.values())
