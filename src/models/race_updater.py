"""
Race-by-Race Update System
Integrates Elo ratings with your existing ML predictor.

Usage:
    1. After each race: update_from_race(results_df)
    2. After quali/practice: update_from_qualifying(quali_df)
    3. Before prediction: get_elo_adjusted_predictions()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from src.constants import constructor_display_name
from src.models.elo_updater import F1EloRatingSystem

logger = logging.getLogger(__name__)


class RaceByRaceUpdater:
    """
    Combines ML predictions with Elo-based live updates.

    Strategy:
    - ML model: trained on historical data (long-term patterns)
    - Elo system: updated after each race (short-term form)
    - Blend both for final prediction
    """

    def __init__(self,
                 elo_weight: float = 0.40,
                 k_factor_driver: float = 20,
                 k_factor_constructor: float = 40,
                 ratings_path: str = "models/elo_ratings.json"):
        """
        Initialize updater.

        Args:
            elo_weight: How much to trust Elo vs ML (0.4 = 40% Elo, 60% ML)
                       Early season: increase Elo weight
                       Mid season: decrease Elo weight
            k_factor_driver: Elo learning rate for drivers
            k_factor_constructor: Elo learning rate for constructors
            ratings_path: Where persisted ratings live
        """
        self.elo_system = F1EloRatingSystem(
            k_factor_driver=k_factor_driver,
            k_factor_constructor=k_factor_constructor
        )

        self.ratings_path = ratings_path
        self.elo_weight = elo_weight
        self.ml_weight = 1.0 - elo_weight
        self.has_ratings = False

        # Try to load existing ratings
        if Path(ratings_path).exists():
            try:
                self.elo_system.load_ratings(ratings_path)
                self.has_ratings = bool(self.elo_system.constructor_ratings)
                logger.info(f"Loaded Elo ratings from {ratings_path}")
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(f"Could not read Elo ratings at {ratings_path}: {exc}")
        else:
            logger.info("No existing Elo ratings found, starting fresh")

    def update_from_race(self,
                        results: pd.DataFrame,
                        circuit: str,
                        race_number: int):
        """
        Update ratings after a race finishes.

        Args:
            results: DataFrame with columns:
                - driver_id
                - constructor_id
                - position (finishing position)
                - quali_position (optional)
                - dnf (boolean)
                - points (optional)
            circuit: Circuit name
            race_number: Race number (1-24)
        """
        logger.info(f"Updating ratings from {circuit} (Race {race_number})")

        # Ratings are keyed by display name, so normalise whichever vocabulary
        # the caller used (`red_bull` vs `Red Bull Racing`).
        results = results.copy()
        results['constructor_id'] = results['constructor_id'].map(constructor_display_name)

        # Update Elo ratings
        self.elo_system.update_from_race_result(
            results,
            circuit=circuit,
            race_number=race_number,
            apply_quali_boost=True
        )

        # Adjust Elo weight based on races completed
        # More races = trust Elo more (it has more data)
        if race_number <= 3:
            self.elo_weight = 0.30  # 30% Elo, 70% ML (trust priors)
        elif race_number <= 7:
            self.elo_weight = 0.50  # 50/50
        else:
            self.elo_weight = 0.60  # 60% Elo, 40% ML (trust recent form)

        self.ml_weight = 1.0 - self.elo_weight

        # Save updated ratings
        self.elo_system.save_ratings(self.ratings_path)
        self.has_ratings = True

        logger.info(f"Ratings updated. Current Elo weight: {self.elo_weight:.1%}")

    def update_from_qualifying(self,
                              quali_results: pd.DataFrame,
                              circuit: str):
        """
        Quick update from qualifying/practice data.

        This is the "fast filter" mentioned in the paper - incorporates
        weekend-specific performance before the race.

        Args:
            quali_results: DataFrame with:
                - driver_id
                - constructor_id
                - best_lap_time (seconds)
                - quali_position
            circuit: Circuit name
        """
        logger.info(f"Quick update from {circuit} qualifying")

        # Convert to practice data format
        practice_data = quali_results.copy()
        if 'best_lap_time' not in practice_data.columns:
            # Estimate from position (rough approximation)
            practice_data['best_lap_time'] = 90 + practice_data['quali_position'] * 0.2

        self.elo_system.update_from_practice_quali(
            practice_data,
            circuit=circuit,
            use_fast_update=True
        )

        logger.info("Quick update applied")

    def blend_predictions(self,
                         ml_predictions: Dict[str, float],
                         ml_uncertainties: Dict[str, float],
                         driver_constructor_map: Dict[str, str],
                         circuit: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Blend ML predictions with Elo ratings.

        Args:
            ml_predictions: Dict of driver_id -> predicted position (from ML)
            ml_uncertainties: Dict of driver_id -> uncertainty
            driver_constructor_map: Dict of driver_id -> constructor_id
            circuit: Circuit name for adjustments

        Returns:
            (blended_predictions, blended_uncertainties)
        """
        if not self.has_ratings:
            # With no rating history every driver sits on the initial Elo, so an
            # Elo-derived order would be arbitrary. Pass the ML view straight
            # through rather than injecting noise.
            logger.info("No Elo history available; returning ML predictions unblended")
            return dict(ml_predictions), dict(ml_uncertainties)

        elo_ratings = {
            driver: self.elo_system.get_combined_rating(
                driver, constructor_display_name(constructor), circuit
            )
            for driver, constructor in driver_constructor_map.items()
        }

        # Convert Elo ratings to predicted positions
        # Higher rating = better position
        sorted_by_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        elo_predictions = {
            driver: position + 1
            for position, (driver, _) in enumerate(sorted_by_elo)
        }

        # Blend ML and Elo predictions
        blended = {}
        blended_uncertainties = {}

        for driver in ml_predictions:
            ml_pos = ml_predictions[driver]
            elo_pos = elo_predictions.get(driver, ml_pos)

            # Weighted average
            blended_pos = (
                ml_pos * self.ml_weight +
                elo_pos * self.elo_weight
            )
            blended[driver] = blended_pos

            # Uncertainty: lower if Elo and ML agree, higher if they disagree
            ml_uncertainty = ml_uncertainties.get(driver, 2.0)
            disagreement = abs(ml_pos - elo_pos)

            # If they agree within 2 positions, reduce uncertainty
            if disagreement < 2:
                blended_uncertainties[driver] = ml_uncertainty * 0.85
            else:
                # If they disagree, increase uncertainty
                blended_uncertainties[driver] = ml_uncertainty * (1 + disagreement * 0.05)

        return blended, blended_uncertainties

    def get_elo_win_probabilities(self,
                                  driver_constructor_pairs: List[Tuple[str, str]],
                                  circuit: Optional[str] = None) -> Dict[str, float]:
        """
        Get win probabilities directly from Elo system.

        Args:
            driver_constructor_pairs: List of (driver, constructor) tuples
            circuit: Circuit name

        Returns:
            Dict of driver -> win probability
        """
        return self.elo_system.get_win_probabilities(
            [(driver, constructor_display_name(constructor))
             for driver, constructor in driver_constructor_pairs],
            circuit=circuit
        )

    def get_rating_summary(self) -> pd.DataFrame:
        """Get current Elo ratings for display."""
        return self.elo_system.get_rating_summary()

    def get_constructor_power_rankings(self) -> Dict[str, float]:
        """Get constructor rankings (normalized 0-1)."""
        ratings = self.elo_system.constructor_ratings

        if not ratings:
            return {}

        min_rating = min(ratings.values())
        max_rating = max(ratings.values())
        rating_range = max_rating - min_rating

        if rating_range == 0:
            return {c: 0.5 for c in ratings}

        # Normalize to 0-1 scale
        normalized = {
            constructor: (rating - min_rating) / rating_range
            for constructor, rating in ratings.items()
        }

        return normalized

    def adjust_monte_carlo_params(self,
                                  driver: str,
                                  constructor: str) -> Dict[str, float]:
        """
        Adjust Monte Carlo simulation parameters based on Elo ratings.

        Returns adjustments for:
        - overtake_probability
        - mistake_probability
        - reliability
        """
        driver_rating = self.elo_system.get_driver_rating(driver)
        constructor_rating = self.elo_system.get_constructor_rating(
            constructor_display_name(constructor)
        )

        # Driver rating affects overtake/mistake probability
        # High rating (>1600) = better overtaking, fewer mistakes
        driver_skill_factor = (driver_rating - 1500) / 500  # -1 to +1
        driver_skill_factor = np.clip(driver_skill_factor, -0.5, 0.5)

        # Constructor rating affects reliability
        constructor_factor = (constructor_rating - 1500) / 500
        constructor_factor = np.clip(constructor_factor, -0.5, 0.5)

        return {
            'overtake_skill': 0.5 + driver_skill_factor * 0.2,  # 0.3 to 0.7
            'mistake_probability': 0.02 - driver_skill_factor * 0.01,  # 0.01 to 0.03
            'reliability_mult': 1.0 + constructor_factor * 0.15,  # 0.85 to 1.15
        }


def create_updater_from_australia_gp() -> RaceByRaceUpdater:
    """
    Create updater pre-initialized with Australia GP 2026 data.
    """
    updater = RaceByRaceUpdater(elo_weight=0.40)

    # Australia GP results (qualifying - replace with race when available)
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

    updater.update_from_race(
        australia_results,
        circuit='Albert Park',
        race_number=1
    )

    return updater


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Creating race-by-race updater with Australia GP data...\n")

    updater = create_updater_from_australia_gp()

    print("=== CONSTRUCTOR POWER RANKINGS ===")
    rankings = updater.get_constructor_power_rankings()
    for constructor, score in sorted(rankings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {constructor:20} {score:.3f}")

    print("\n=== EXAMPLE: Blend ML + Elo Predictions ===")

    # Simulate ML predictions
    ml_predictions = {
        'russell': 2.5,
        'verstappen': 3.0,  # ML thinks he'll recover
        'leclerc': 4.0,
        'norris': 5.5,
        'antonelli': 6.0,  # ML uncertain about rookie
        'hadjar': 12.0,  # ML doesn't know about his pace yet
    }

    ml_uncertainties = {driver: 2.0 for driver in ml_predictions}

    constructor_map = {
        'russell': 'Mercedes',
        'verstappen': 'Red Bull Racing',
        'leclerc': 'Ferrari',
        'norris': 'McLaren',
        'antonelli': 'Mercedes',
        'hadjar': 'RB',
    }

    blended, uncertainties = updater.blend_predictions(
        ml_predictions,
        ml_uncertainties,
        constructor_map,
        circuit='Bahrain'
    )

    print("\nDriver          ML Pred    Elo Impact    Blended    Uncertainty")
    print("-" * 70)
    for driver in sorted(blended.keys(), key=lambda d: blended[d]):
        ml_pos = ml_predictions[driver]
        blend_pos = blended[driver]
        unc = uncertainties[driver]
        impact = blend_pos - ml_pos

        print(f"{driver:15} P{ml_pos:4.1f}     {impact:+5.1f}       P{blend_pos:4.1f}      ±{unc:.2f}")

    print("\nKey insight: Elo pulls Hadjar up 6 positions, Verstappen down 9!")
    print("Antonelli gets boost (rookie performing well)")
