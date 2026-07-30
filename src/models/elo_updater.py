"""
Elo-style Rating System for F1 2026
Separates driver skill from constructor performance with race-by-race updates.

Based on research:
- Modified Elo for motorsport (SIAM)
- Bayesian updating with quick online adjustments
- Separate constructor and driver components
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime


class F1EloRatingSystem:
    """
    Modified Elo rating system for F1 that separates:
    - Constructor (car) performance
    - Driver skill
    - Circuit-specific adjustments

    Updates after each race with fast online learning.
    """

    def __init__(self,
                 initial_driver_rating: float = 1500,
                 initial_constructor_rating: float = 1500,
                 k_factor_driver: float = 20,
                 k_factor_constructor: float = 40,
                 decay_factor: float = 0.95):
        """
        Initialize Elo rating system.

        Args:
            initial_driver_rating: Starting Elo for all drivers
            initial_constructor_rating: Starting Elo for all constructors
            k_factor_driver: Learning rate for drivers (lower = slower updates)
            k_factor_constructor: Learning rate for constructors (higher = faster)
            decay_factor: Regression to mean between seasons
        """
        self.driver_ratings: Dict[str, float] = {}
        self.constructor_ratings: Dict[str, float] = {}

        self.initial_driver_rating = initial_driver_rating
        self.initial_constructor_rating = initial_constructor_rating

        # K-factors control how fast ratings update
        # Constructor changes faster (new regulations, upgrades)
        # Driver changes slower (skill is more stable)
        self.k_driver = k_factor_driver
        self.k_constructor = k_factor_constructor

        self.decay_factor = decay_factor

        # Track rating history
        self.rating_history: List[Dict] = []
        self.race_counter = 0

        # Circuit-specific adjustments (learned from data)
        self.circuit_adjustments: Dict[str, Dict[str, float]] = {}

    def get_driver_rating(self, driver: str) -> float:
        """Get current driver rating, initializing if needed."""
        if driver not in self.driver_ratings:
            self.driver_ratings[driver] = self.initial_driver_rating
        return self.driver_ratings[driver]

    def get_constructor_rating(self, constructor: str) -> float:
        """Get current constructor rating, initializing if needed."""
        if constructor not in self.constructor_ratings:
            self.constructor_ratings[constructor] = self.initial_constructor_rating
        return self.constructor_ratings[constructor]

    def get_combined_rating(self,
                           driver: str,
                           constructor: str,
                           circuit: Optional[str] = None,
                           driver_weight: float = 0.30,
                           constructor_weight: float = 0.70) -> float:
        """
        Get combined rating for driver + constructor pair.

        Research shows constructor explains ~80% of variance in hybrid era,
        so we weight it at 70% and driver at 30%.

        Args:
            driver: Driver identifier
            constructor: Team identifier
            circuit: Circuit name for adjustments
            driver_weight: Weight for driver component (0.3 = 30%)
            constructor_weight: Weight for constructor component (0.7 = 70%)

        Returns:
            Combined Elo rating
        """
        driver_elo = self.get_driver_rating(driver)
        constructor_elo = self.get_constructor_rating(constructor)

        combined = (driver_elo * driver_weight +
                   constructor_elo * constructor_weight)

        # Apply circuit-specific adjustment if available
        if circuit and circuit in self.circuit_adjustments:
            if constructor in self.circuit_adjustments[circuit]:
                combined += self.circuit_adjustments[circuit][constructor]

        return combined

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score (win probability) using Elo formula.

        Args:
            rating_a: Elo rating of driver A
            rating_b: Elo rating of driver B

        Returns:
            Probability that A finishes ahead of B (0-1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_from_race_result(self,
                                race_results: pd.DataFrame,
                                circuit: str,
                                race_number: int,
                                apply_quali_boost: bool = True):
        """
        Update ratings based on actual race results.

        Uses pairwise comparisons: for each pair of finishers,
        update based on who finished ahead.

        Args:
            race_results: DataFrame with columns:
                - driver_id
                - constructor_id
                - position (finishing position)
                - quali_position (optional, for qualifying boost)
                - dnf (boolean, True if didn't finish)
            circuit: Circuit name
            race_number: Race number in season
            apply_quali_boost: Whether to boost constructor rating from quali
        """
        self.race_counter += 1

        # Sort by finishing position
        results = race_results.sort_values('position').copy()

        # Store pre-update ratings for history
        pre_ratings = {
            'drivers': self.driver_ratings.copy(),
            'constructors': self.constructor_ratings.copy()
        }

        # Pairwise updates
        for i, row_i in results.iterrows():
            driver_i = row_i['driver_id']
            constructor_i = row_i['constructor_id']
            position_i = row_i['position']
            dnf_i = row_i.get('dnf', False)

            # Get current ratings
            combined_i = self.get_combined_rating(driver_i, constructor_i, circuit)

            # Compare against all drivers who finished behind
            for j, row_j in results.iterrows():
                if i == j:
                    continue

                driver_j = row_j['driver_id']
                constructor_j = row_j['constructor_id']
                position_j = row_j['position']
                dnf_j = row_j.get('dnf', False)

                # Skip if didn't actually compete
                if position_i >= 20 and position_j >= 20:
                    continue

                combined_j = self.get_combined_rating(driver_j, constructor_j, circuit)

                # Actual result: did i beat j?
                if position_i < position_j:
                    actual_score = 1.0
                elif position_i > position_j:
                    actual_score = 0.0
                else:
                    actual_score = 0.5  # Tie

                # Expected score (probability)
                expected = self.expected_score(combined_i, combined_j)

                # Calculate error
                error = actual_score - expected

                # DNF penalty: if you DNF'd, larger update
                k_mult_i = 1.5 if dnf_i else 1.0
                k_mult_j = 1.5 if dnf_j else 1.0

                # Update driver ratings (30% of update)
                driver_update_i = self.k_driver * k_mult_i * error * 0.3
                driver_update_j = self.k_driver * k_mult_j * (-error) * 0.3

                self.driver_ratings[driver_i] = self.get_driver_rating(driver_i) + driver_update_i
                self.driver_ratings[driver_j] = self.get_driver_rating(driver_j) + driver_update_j

                # Update constructor ratings (70% of update)
                constructor_update_i = self.k_constructor * k_mult_i * error * 0.7
                constructor_update_j = self.k_constructor * k_mult_j * (-error) * 0.7

                self.constructor_ratings[constructor_i] = (
                    self.get_constructor_rating(constructor_i) + constructor_update_i
                )
                self.constructor_ratings[constructor_j] = (
                    self.get_constructor_rating(constructor_j) + constructor_update_j
                )

        # Qualifying boost: if apply_quali_boost, reward constructors for qualifying performance
        if apply_quali_boost and 'quali_position' in results.columns:
            self._apply_qualifying_boost(results, circuit)

        # Store rating history
        self.rating_history.append({
            'race_number': race_number,
            'circuit': circuit,
            'date': datetime.now().isoformat(),
            'pre_ratings': pre_ratings,
            'post_ratings': {
                'drivers': self.driver_ratings.copy(),
                'constructors': self.constructor_ratings.copy()
            }
        })

    def _apply_qualifying_boost(self, results: pd.DataFrame, circuit: str):
        """
        Small boost to constructor rating based on qualifying performance.
        Qualifying contains strong signal for car performance.
        """
        if 'quali_position' not in results.columns:
            return

        # Calculate quali vs race delta
        for _, row in results.iterrows():
            constructor = row['constructor_id']
            quali_pos = row.get('quali_position', 20)
            race_pos = row['position']

            # If qualified well, small constructor boost
            if quali_pos <= 10:
                boost = (11 - quali_pos) * 2  # P1 gets +20, P10 gets +2
                self.constructor_ratings[constructor] = (
                    self.get_constructor_rating(constructor) + boost
                )

    def update_from_practice_quali(self,
                                   practice_data: pd.DataFrame,
                                   circuit: str,
                                   use_fast_update: bool = True):
        """
        Quick update from practice/qualifying data.

        This is a "fast filter" update between races - uses lap times
        to adjust constructor ratings for the specific circuit.

        Args:
            practice_data: DataFrame with:
                - driver_id
                - constructor_id
                - best_lap_time (seconds)
                - sector_1, sector_2, sector_3 (optional)
            circuit: Circuit name
            use_fast_update: Use Kalman-filter-style quick update
        """
        if circuit not in self.circuit_adjustments:
            self.circuit_adjustments[circuit] = {}

        # Find fastest lap
        fastest_time = practice_data['best_lap_time'].min()

        for _, row in practice_data.iterrows():
            constructor = row['constructor_id']
            lap_time = row['best_lap_time']

            # Calculate delta to fastest
            delta = lap_time - fastest_time

            # Convert to rating adjustment
            # 0.1s gap = ~15 Elo points
            adjustment = -delta * 150

            # Apply to circuit-specific adjustments
            if constructor not in self.circuit_adjustments[circuit]:
                self.circuit_adjustments[circuit][constructor] = 0

            # Fast Kalman-filter-style update: blend with existing
            if use_fast_update:
                alpha = 0.7  # Trust new data at 70%
                self.circuit_adjustments[circuit][constructor] = (
                    alpha * adjustment +
                    (1 - alpha) * self.circuit_adjustments[circuit][constructor]
                )
            else:
                self.circuit_adjustments[circuit][constructor] = adjustment

    def get_win_probabilities(self,
                             driver_constructor_pairs: List[Tuple[str, str]],
                             circuit: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate win probability for each driver-constructor pair.

        Uses softmax over combined Elo ratings.

        Args:
            driver_constructor_pairs: List of (driver_id, constructor_id) tuples
            circuit: Circuit name for adjustments

        Returns:
            Dict mapping driver_id to win probability
        """
        # Get combined ratings
        ratings = {}
        for driver, constructor in driver_constructor_pairs:
            rating = self.get_combined_rating(driver, constructor, circuit)
            ratings[driver] = rating

        # Convert to probabilities using softmax
        rating_array = np.array(list(ratings.values()))
        # Scale down for numerical stability
        scaled_ratings = (rating_array - rating_array.mean()) / 100
        exp_ratings = np.exp(scaled_ratings)
        probabilities = exp_ratings / exp_ratings.sum()

        # Map back to drivers
        win_probs = {}
        for i, driver in enumerate(ratings.keys()):
            win_probs[driver] = probabilities[i]

        return win_probs

    def apply_season_decay(self):
        """
        Apply regression to mean at end of season.
        New regulations = more uncertainty, so ratings move toward baseline.
        """
        for driver in self.driver_ratings:
            self.driver_ratings[driver] = (
                self.driver_ratings[driver] * self.decay_factor +
                self.initial_driver_rating * (1 - self.decay_factor)
            )

        for constructor in self.constructor_ratings:
            self.constructor_ratings[constructor] = (
                self.constructor_ratings[constructor] * self.decay_factor +
                self.initial_constructor_rating * (1 - self.decay_factor)
            )

    def save_ratings(self, path: str = "models/elo_ratings.json"):
        """Save current ratings to file."""
        output = {
            'driver_ratings': self.driver_ratings,
            'constructor_ratings': self.constructor_ratings,
            'circuit_adjustments': self.circuit_adjustments,
            'race_counter': self.race_counter,
            'metadata': {
                'k_driver': self.k_driver,
                'k_constructor': self.k_constructor,
                'last_updated': datetime.now().isoformat()
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)

    def load_ratings(self, path: str = "models/elo_ratings.json"):
        """Load ratings from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.driver_ratings = data['driver_ratings']
        self.constructor_ratings = data['constructor_ratings']
        self.circuit_adjustments = data.get('circuit_adjustments', {})
        self.race_counter = data.get('race_counter', 0)

    def get_rating_summary(self) -> pd.DataFrame:
        """Get current ratings as DataFrame for display."""
        driver_data = [
            {'entity': driver, 'rating': rating, 'type': 'driver'}
            for driver, rating in self.driver_ratings.items()
        ]
        constructor_data = [
            {'entity': constructor, 'rating': rating, 'type': 'constructor'}
            for constructor, rating in self.constructor_ratings.items()
        ]

        df = pd.DataFrame(driver_data + constructor_data)
        return df.sort_values('rating', ascending=False)


def initialize_from_australia_gp() -> F1EloRatingSystem:
    """
    Initialize Elo system with Australia GP 2026 results as first data point.
    """
    elo = F1EloRatingSystem(
        k_factor_driver=20,
        k_factor_constructor=40  # Constructor updates faster
    )

    # Australia GP qualifying results
    australia_results = pd.DataFrame([
        {'driver_id': 'russell', 'constructor_id': 'Mercedes', 'position': 1, 'quali_position': 1, 'dnf': False},
        {'driver_id': 'antonelli', 'constructor_id': 'Mercedes', 'position': 2, 'quali_position': 2, 'dnf': False},
        {'driver_id': 'hadjar', 'constructor_id': 'RB', 'position': 3, 'quali_position': 3, 'dnf': False},
        {'driver_id': 'leclerc', 'constructor_id': 'Ferrari', 'position': 4, 'quali_position': 4, 'dnf': False},
        {'driver_id': 'piastri', 'constructor_id': 'McLaren', 'position': 5, 'quali_position': 5, 'dnf': False},
        {'driver_id': 'norris', 'constructor_id': 'McLaren', 'position': 6, 'quali_position': 6, 'dnf': False},
        {'driver_id': 'hamilton', 'constructor_id': 'Ferrari', 'position': 7, 'quali_position': 7, 'dnf': False},
        {'driver_id': 'lindblad', 'constructor_id': 'RB', 'position': 8, 'quali_position': 8, 'dnf': False},
        {'driver_id': 'lawson', 'constructor_id': 'RB', 'position': 9, 'quali_position': 9, 'dnf': False},
        {'driver_id': 'bortoleto', 'constructor_id': 'Audi (Sauber)', 'position': 10, 'quali_position': 10, 'dnf': True},
        {'driver_id': 'hulkenberg', 'constructor_id': 'Audi (Sauber)', 'position': 11, 'quali_position': 11, 'dnf': False},
        {'driver_id': 'bearman', 'constructor_id': 'Haas', 'position': 12, 'quali_position': 12, 'dnf': False},
        {'driver_id': 'ocon', 'constructor_id': 'Haas', 'position': 13, 'quali_position': 13, 'dnf': False},
        {'driver_id': 'gasly', 'constructor_id': 'Alpine', 'position': 14, 'quali_position': 14, 'dnf': False},
        {'driver_id': 'albon', 'constructor_id': 'Williams', 'position': 15, 'quali_position': 15, 'dnf': False},
        {'driver_id': 'colapinto', 'constructor_id': 'Alpine', 'position': 16, 'quali_position': 16, 'dnf': False},
        {'driver_id': 'alonso', 'constructor_id': 'Aston Martin', 'position': 17, 'quali_position': 17, 'dnf': False},
        {'driver_id': 'verstappen', 'constructor_id': 'Red Bull Racing', 'position': 20, 'quali_position': 20, 'dnf': True},
        {'driver_id': 'sainz', 'constructor_id': 'Williams', 'position': 21, 'quali_position': 21, 'dnf': True},
        {'driver_id': 'stroll', 'constructor_id': 'Aston Martin', 'position': 22, 'quali_position': 22, 'dnf': True},
    ])

    elo.update_from_race_result(
        australia_results,
        circuit='Albert Park',
        race_number=1,
        apply_quali_boost=True
    )

    return elo


if __name__ == "__main__":
    # Example: Initialize with Australia GP
    print("Initializing Elo system with Australia GP 2026 data...\n")

    elo = initialize_from_australia_gp()

    print("=== UPDATED ELO RATINGS (After Australia GP) ===\n")

    print("Constructor Ratings:")
    constructor_ratings = sorted(
        elo.constructor_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for constructor, rating in constructor_ratings:
        print(f"  {constructor:20} {rating:.1f}")

    print("\nTop 10 Driver Ratings:")
    driver_ratings = sorted(
        elo.driver_ratings.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for driver, rating in driver_ratings:
        print(f"  {driver:20} {rating:.1f}")

    # Example win probability calculation
    print("\n=== WIN PROBABILITIES (Next Race - Bahrain) ===")

    pairs = [
        ('russell', 'Mercedes'),
        ('antonelli', 'Mercedes'),
        ('verstappen', 'Red Bull Racing'),
        ('leclerc', 'Ferrari'),
        ('norris', 'McLaren'),
        ('hadjar', 'RB'),
        ('bortoleto', 'Audi (Sauber)'),
    ]

    win_probs = elo.get_win_probabilities(pairs, circuit='Bahrain')

    sorted_probs = sorted(win_probs.items(), key=lambda x: x[1], reverse=True)
    for driver, prob in sorted_probs:
        print(f"  {driver:15} {prob*100:5.1f}%")

    # Save ratings
    elo.save_ratings()
    print("\n✅ Ratings saved to models/elo_ratings.json")
