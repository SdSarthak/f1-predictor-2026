"""
F1 Race Predictor - Monte Carlo Race Simulation
Runs probabilistic race simulations with random events.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RaceEvent:
    """Represents a random event during a race."""
    lap: int
    event_type: str  # 'safety_car', 'red_flag', 'dnf', 'collision', 'rain'
    affected_drivers: List[str]
    impact: Dict[str, Any]


class MonteCarloSimulator:
    """
    Monte Carlo race simulator for probabilistic predictions.
    Runs thousands of race simulations to estimate win probabilities.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.mc_config = self.config.get('monte_carlo', {})
        self.num_simulations = self.mc_config.get('num_simulations', 10000)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration, falling back to the built-in event probabilities."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Could not load {config_path}: {exc}. Using defaults.")
            return {}
    
    def simulate_race(self, 
                     predicted_positions: Dict[str, float],
                     position_uncertainty: Dict[str, float],
                     grid_positions: Dict[str, int],
                     reliability_scores: Dict[str, float],
                     race_laps: int = 57,
                     circuit_type: str = 'normal') -> Dict[str, Any]:
        """
        Simulate a single race.
        
        Args:
            predicted_positions: Dict of driver_id -> predicted position
            position_uncertainty: Dict of driver_id -> uncertainty (std)
            grid_positions: Dict of driver_id -> starting grid position
            reliability_scores: Dict of driver_id -> reliability (0-1)
            race_laps: Number of laps in the race
            circuit_type: 'street', 'high_speed', 'high_downforce'
            
        Returns:
            Dict with final positions and events
        """
        drivers = list(predicted_positions.keys())
        
        # Initialize positions from predictions with uncertainty
        positions = {}
        for driver in drivers:
            base_pos = predicted_positions[driver]
            uncertainty = position_uncertainty.get(driver, 1.5)
            # Sample from normal distribution centered on predicted position
            positions[driver] = np.random.normal(base_pos, uncertainty)
        
        events = []
        dnf_drivers = set()
        
        # Simulate first lap chaos
        if np.random.random() < self.mc_config.get('first_lap_incident_probability', 0.15):
            # First lap incident affects back of grid more
            affected = self._simulate_first_lap_incident(drivers, grid_positions)
            for driver in affected:
                dnf_drivers.add(driver)
                events.append(RaceEvent(
                    lap=1,
                    event_type='first_lap_collision',
                    affected_drivers=[driver],
                    impact={'dnf': True}
                ))
        
        # Simulate race lap by lap (simplified to key events)
        for lap in range(2, race_laps + 1):
            # Safety car probability
            if np.random.random() < self._get_safety_car_prob(circuit_type):
                events.append(RaceEvent(
                    lap=lap,
                    event_type='safety_car',
                    affected_drivers=drivers,
                    impact={'position_compression': 0.3}
                ))
                # Compress field
                positions = self._compress_field(positions, dnf_drivers, factor=0.3)
            
            # Red flag probability
            if np.random.random() < self.mc_config.get('red_flag_probability', 0.08):
                events.append(RaceEvent(
                    lap=lap,
                    event_type='red_flag',
                    affected_drivers=drivers,
                    impact={'reset': True}
                ))
                # Reset to closer gaps
                positions = self._compress_field(positions, dnf_drivers, factor=0.5)
            
            # DNF probabilities per driver
            for driver in drivers:
                if driver in dnf_drivers:
                    continue
                    
                reliability = reliability_scores.get(driver, 0.9)
                # DNF probability inversely proportional to reliability
                dnf_prob = (1 - reliability) / race_laps
                
                if np.random.random() < dnf_prob:
                    dnf_drivers.add(driver)
                    events.append(RaceEvent(
                        lap=lap,
                        event_type='dnf',
                        affected_drivers=[driver],
                        impact={'cause': 'mechanical'}
                    ))
        
        # Calculate final positions
        final_positions = self._calculate_final_positions(positions, dnf_drivers)
        
        return {
            'final_positions': final_positions,
            'dnf_drivers': list(dnf_drivers),
            'events': events,
        }
    
    def _get_safety_car_prob(self, circuit_type: str) -> float:
        """Get safety car probability based on circuit type."""
        base_prob = self.mc_config.get('safety_car_probability', 0.35) / 57  # Per lap
        
        multipliers = {
            'street': 1.5,
            'high_speed': 0.8,
            'high_downforce': 1.2,
            'normal': 1.0,
        }
        
        return base_prob * multipliers.get(circuit_type, 1.0)
    
    def _simulate_first_lap_incident(self, 
                                    drivers: List[str], 
                                    grid_positions: Dict[str, int]) -> List[str]:
        """Simulate first lap chaos - typically affects midfield/back."""
        # Higher chance of incident for drivers starting P10-P20
        at_risk = [d for d in drivers if grid_positions.get(d, 20) > 8]
        
        num_affected = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        
        if len(at_risk) >= num_affected:
            return list(np.random.choice(at_risk, num_affected, replace=False))
        return []
    
    def _compress_field(self, 
                       positions: Dict[str, float],
                       dnf_drivers: set,
                       factor: float) -> Dict[str, float]:
        """Compress the field during safety car/red flag."""
        active_drivers = {d: p for d, p in positions.items() if d not in dnf_drivers}
        
        if not active_drivers:
            return positions
            
        # Sort by position
        sorted_drivers = sorted(active_drivers.items(), key=lambda x: x[1])
        
        # Compress gaps
        new_positions = {}
        for i, (driver, _) in enumerate(sorted_drivers, 1):
            # Add some randomness within compressed gaps
            compressed_pos = i + np.random.normal(0, 0.1 * factor)
            new_positions[driver] = compressed_pos
        
        # Add back DNF drivers with high position
        for driver in dnf_drivers:
            new_positions[driver] = 25
            
        return new_positions
    
    def _calculate_final_positions(self, 
                                   positions: Dict[str, float],
                                   dnf_drivers: set) -> Dict[str, int]:
        """Convert floating positions to final integer positions."""
        # Active drivers
        active = {d: p for d, p in positions.items() if d not in dnf_drivers}
        sorted_active = sorted(active.items(), key=lambda x: x[1])
        
        final = {}
        for i, (driver, _) in enumerate(sorted_active, 1):
            final[driver] = i
        
        # DNF drivers get positions after last finisher
        next_pos = len(active) + 1
        for driver in dnf_drivers:
            final[driver] = next_pos
            next_pos += 1
            
        return final
    
    def run_monte_carlo(self,
                       predicted_positions: Dict[str, float],
                       position_uncertainty: Dict[str, float],
                       grid_positions: Dict[str, int],
                       reliability_scores: Dict[str, float],
                       race_laps: int = 57,
                       circuit_type: str = 'normal',
                       n_simulations: int = None) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation.
        
        Returns:
            Dict containing:
                - win_probabilities: Probability of each driver winning
                - podium_probabilities: Probability of finishing top 3
                - points_probabilities: Probability of finishing top 10
                - expected_positions: Mean finishing position
                - position_distributions: Full distribution of finishes
        """
        n_simulations = n_simulations or self.num_simulations
        
        drivers = list(predicted_positions.keys())
        
        # Results storage
        wins = defaultdict(int)
        podiums = defaultdict(int)
        points_finishes = defaultdict(int)
        position_sums = defaultdict(float)
        position_lists = defaultdict(list)
        dnf_counts = defaultdict(int)
        
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")
        
        for sim in range(n_simulations):
            if sim % 1000 == 0 and sim > 0:
                logger.info(f"  Completed {sim} simulations")
            
            result = self.simulate_race(
                predicted_positions,
                position_uncertainty,
                grid_positions,
                reliability_scores,
                race_laps,
                circuit_type
            )
            
            final_positions = result['final_positions']
            
            for driver in drivers:
                pos = final_positions.get(driver, 20)
                position_sums[driver] += pos
                position_lists[driver].append(pos)
                
                if pos == 1:
                    wins[driver] += 1
                if pos <= 3:
                    podiums[driver] += 1
                if pos <= 10:
                    points_finishes[driver] += 1
                if driver in result['dnf_drivers']:
                    dnf_counts[driver] += 1
        
        # Calculate probabilities
        results = {
            'win_probabilities': {},
            'podium_probabilities': {},
            'points_probabilities': {},
            'expected_positions': {},
            'position_std': {},
            'dnf_probability': {},
            'position_distributions': {},
        }
        
        for driver in drivers:
            results['win_probabilities'][driver] = wins[driver] / n_simulations
            results['podium_probabilities'][driver] = podiums[driver] / n_simulations
            results['points_probabilities'][driver] = points_finishes[driver] / n_simulations
            results['expected_positions'][driver] = position_sums[driver] / n_simulations
            results['position_std'][driver] = np.std(position_lists[driver])
            results['dnf_probability'][driver] = dnf_counts[driver] / n_simulations
            results['position_distributions'][driver] = position_lists[driver]
        
        # Sort by win probability
        sorted_by_win = sorted(
            results['win_probabilities'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.info("\nMonte Carlo Results (Top 10 by Win Probability):")
        logger.info("-" * 60)
        for driver, win_prob in sorted_by_win[:10]:
            exp_pos = results['expected_positions'][driver]
            podium_prob = results['podium_probabilities'][driver]
            logger.info(f"{driver:20} | Win: {win_prob*100:5.1f}% | "
                       f"Podium: {podium_prob*100:5.1f}% | "
                       f"Exp Pos: {exp_pos:.1f}")
        
        return results
    
    def get_confidence_intervals(self, 
                                position_distributions: Dict[str, List[int]],
                                confidence: float = 0.9) -> Dict[str, Tuple[int, int]]:
        """
        Calculate confidence intervals for finishing positions.
        
        Returns:
            Dict of driver_id -> (lower_bound, upper_bound)
        """
        intervals = {}
        alpha = 1 - confidence
        
        for driver, positions in position_distributions.items():
            lower = np.percentile(positions, alpha/2 * 100)
            upper = np.percentile(positions, (1 - alpha/2) * 100)
            intervals[driver] = (int(np.ceil(lower)), int(np.floor(upper)))
        
        return intervals
    
    def simulate_championship(self,
                             race_results: List[Dict[str, Any]],
                             remaining_races: int = 0,
                             n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Simulate championship outcomes based on race simulations.
        
        Args:
            race_results: List of MC results for each remaining race
            remaining_races: Number of races left
            n_simulations: Number of championship simulations
            
        Returns:
            Championship probability distributions
        """
        # Points system (2025/2026)
        points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }
        
        # For each simulation, sample from position distributions
        championship_counts = defaultdict(int)
        
        for _ in range(n_simulations):
            total_points = defaultdict(int)
            
            for race_result in race_results:
                for driver, positions in race_result['position_distributions'].items():
                    # Sample a finish position
                    finish = np.random.choice(positions)
                    total_points[driver] += points_system.get(finish, 0)
            
            # Find winner
            winner = max(total_points.items(), key=lambda x: x[1])[0]
            championship_counts[winner] += 1
        
        # Calculate probabilities
        championship_probs = {
            driver: count / n_simulations 
            for driver, count in championship_counts.items()
        }
        
        return {
            'championship_probabilities': championship_probs,
            'sorted_probabilities': sorted(
                championship_probs.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        }


def run_simulation(predicted_positions: Dict[str, float],
                  uncertainty: Dict[str, float],
                  grid: Dict[str, int],
                  reliability: Dict[str, float],
                  n_simulations: int = 10000) -> Dict[str, Any]:
    """
    Convenience function to run Monte Carlo simulation.
    """
    simulator = MonteCarloSimulator()
    return simulator.run_monte_carlo(
        predicted_positions,
        uncertainty,
        grid,
        reliability,
        n_simulations=n_simulations
    )


