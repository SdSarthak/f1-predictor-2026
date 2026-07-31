"""Monte Carlo race simulation."""

import numpy as np
import pytest

from src.simulation.monte_carlo import MonteCarloSimulator

DRIVERS = [f"driver_{i:02d}" for i in range(20)]


@pytest.fixture
def simulator(config_path):
    return MonteCarloSimulator(config_path, seed=20260801)


@pytest.fixture
def race_inputs():
    predicted = {driver: float(i + 1) for i, driver in enumerate(DRIVERS)}
    uncertainty = {driver: 1.5 for driver in DRIVERS}
    grid = {driver: i + 1 for i, driver in enumerate(DRIVERS)}
    reliability = {driver: 0.92 for driver in DRIVERS}
    return predicted, uncertainty, grid, reliability


@pytest.fixture
def mc_results(simulator, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs
    return simulator.run_monte_carlo(
        predicted, uncertainty, grid, reliability,
        race_laps=57, circuit_type='normal', n_simulations=300, seed=7,
    )


def test_single_race_assigns_every_driver_a_unique_position(simulator, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs

    result = simulator.simulate_race(predicted, uncertainty, grid, reliability)
    final = result['final_positions']

    assert set(final) == set(DRIVERS)
    assert sorted(final.values()) == list(range(1, len(DRIVERS) + 1))


def test_dnf_drivers_are_classified_behind_every_finisher(simulator, race_inputs):
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
    predicted, uncertainty, grid, _ = race_inputs

    reliable = simulator.run_monte_carlo(
        predicted, uncertainty, grid, {d: 0.99 for d in DRIVERS},
        n_simulations=150, seed=11)
    fragile = simulator.run_monte_carlo(
        predicted, uncertainty, grid, {d: 0.60 for d in DRIVERS},
        n_simulations=150, seed=11)

    assert (sum(fragile['dnf_probability'].values())
            > sum(reliable['dnf_probability'].values()))


def test_championship_simulation_returns_a_distribution(simulator, mc_results):
    outcome = simulator.simulate_championship(
        [mc_results, mc_results], n_simulations=100, seed=5)

    probabilities = outcome['championship_probabilities']
    assert probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert outcome['sorted_probabilities'][0][1] == max(probabilities.values())


def test_the_same_seed_reproduces_the_same_run(config_path, race_inputs):
    """A published prediction has to be reproducible."""
    predicted, uncertainty, grid, reliability = race_inputs

    first = MonteCarloSimulator(config_path).run_monte_carlo(
        predicted, uncertainty, grid, reliability, n_simulations=120, seed=1234)
    second = MonteCarloSimulator(config_path).run_monte_carlo(
        predicted, uncertainty, grid, reliability, n_simulations=120, seed=1234)

    assert first['win_probabilities'] == second['win_probabilities']
    assert first['expected_positions'] == second['expected_positions']


def test_different_seeds_give_different_runs(config_path, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs

    simulator = MonteCarloSimulator(config_path)
    first = simulator.run_monte_carlo(
        predicted, uncertainty, grid, reliability, n_simulations=120, seed=1)
    second = simulator.run_monte_carlo(
        predicted, uncertainty, grid, reliability, n_simulations=120, seed=2)

    assert first['expected_positions'] != second['expected_positions']


def test_the_simulator_does_not_touch_the_global_numpy_state(simulator, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs

    np.random.seed(99)
    before = np.random.random()

    np.random.seed(99)
    simulator.run_monte_carlo(predicted, uncertainty, grid, reliability,
                              n_simulations=20, seed=3)
    after = np.random.random()

    assert before == after


def test_red_flags_are_a_per_race_probability(simulator, race_inputs):
    """
    `red_flag_probability: 0.08` is documented per race but was drawn every
    lap, which put a red flag in 99% of simulated races.
    """
    predicted, uncertainty, grid, reliability = race_inputs
    reliability = {driver: 1.0 for driver in reliability}  # no retirements

    configured = simulator.mc_config.get('red_flag_probability', 0.08)
    races_with_a_red_flag = 0
    trials = 400
    for _ in range(trials):
        result = simulator.simulate_race(predicted, uncertainty, grid, reliability,
                                         race_laps=57)
        if any(event.event_type == 'red_flag' for event in result['events']):
            races_with_a_red_flag += 1

    observed = races_with_a_red_flag / trials
    assert abs(observed - configured) < 0.05, observed


def test_safety_car_rate_does_not_scale_with_race_distance(simulator):
    """Monaco used to be 37% more likely to be neutralised than Spa."""
    short = simulator._get_safety_car_prob('normal', race_laps=44)
    long = simulator._get_safety_car_prob('normal', race_laps=78)

    def per_race(per_lap, laps):
        return 1 - (1 - per_lap) ** (laps - 1)

    assert per_race(short, 44) == pytest.approx(per_race(long, 78), abs=1e-9)


def test_dnf_rate_matches_the_configured_reliability(simulator, race_inputs):
    predicted, uncertainty, grid, _ = race_inputs
    reliability = {driver: 0.80 for driver in DRIVERS}

    results = simulator.run_monte_carlo(
        predicted, uncertainty, grid, reliability,
        race_laps=57, n_simulations=400, seed=17)

    mean_dnf = np.mean(list(results['dnf_probability'].values()))
    # 0.20 mechanical + a share of the first-lap incident, so a band rather
    # than a point value - but nowhere near the old distance-dependent rate.
    assert 0.18 < mean_dnf < 0.32, mean_dnf


def test_retirements_are_classified_by_how_far_they_got(simulator):
    positions = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    dnf_laps = {'b': 40, 'c': 5}

    final = simulator._calculate_final_positions(positions, dnf_laps)

    assert final['a'] == 1
    assert final['b'] == 2   # retired later, classified ahead
    assert final['c'] == 3


def test_classification_is_independent_of_dict_order(simulator):
    forwards = simulator._calculate_final_positions(
        {'a': 1.0, 'b': 2.0, 'c': 3.0}, {'b': 10, 'c': 10})
    backwards = simulator._calculate_final_positions(
        {'c': 3.0, 'b': 2.0, 'a': 1.0}, {'c': 10, 'b': 10})

    assert forwards == backwards


def test_a_short_field_still_gets_first_lap_incidents(simulator):
    """Fewer at-risk cars than the drawn count used to return nobody."""
    drivers = ['a', 'b']
    grid = {'a': 18, 'b': 19}

    outcomes = [len(simulator._simulate_first_lap_incident(drivers, grid))
                for _ in range(200)]

    assert max(outcomes) <= len(drivers)
    assert min(outcomes) >= 1


def test_simulation_inputs_are_validated(simulator, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs

    with pytest.raises(ValueError, match="No drivers"):
        simulator.run_monte_carlo({}, {}, {}, {})

    with pytest.raises(ValueError, match="at least 1"):
        simulator.run_monte_carlo(predicted, uncertainty, grid, reliability,
                                  n_simulations=0)

    with pytest.raises(ValueError, match="race_laps"):
        simulator.simulate_race(predicted, uncertainty, grid, reliability, race_laps=0)

    with pytest.raises(ValueError, match="confidence"):
        simulator.get_confidence_intervals({'a': [1, 2, 3]}, confidence=1.5)

    with pytest.raises(ValueError, match="position_distributions"):
        simulator.simulate_championship([])


def test_a_single_lap_race_still_produces_a_full_classification(simulator, race_inputs):
    predicted, uncertainty, grid, reliability = race_inputs

    result = simulator.simulate_race(predicted, uncertainty, grid, reliability,
                                     race_laps=1)

    assert sorted(result['final_positions'].values()) == list(range(1, len(DRIVERS) + 1))
