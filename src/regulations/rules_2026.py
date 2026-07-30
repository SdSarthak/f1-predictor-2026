"""
F1 2026 Regulation Adjustments Module
Handles special weighting and uncertainty for the 2026 regulation reset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import yaml
import logging

from src.constants import GRID_2026, TEAM_ENGINE_2026, constructor_display_name
from src.data.testing_data_2026 import Testing2026Data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Regulations2026:
    """
    Handles 2026-specific adjustments for predictions.
    
    Key 2026 changes:
    - New aerodynamic regulations (Active Aero / movable wings)
    - 50/50 Electric/ICE power split
    - Weight reduction (-30kg)
    - Narrower tires
    - New engine manufacturers (Audi, Red Bull Powertrains-Ford)
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.regs_config = self.config.get('regulations_2026', {})
        self.testing_data = Testing2026Data()
        
        # 2026 Team-Engine mapping
        self.team_engine_2026 = dict(TEAM_ENGINE_2026)


        # Historical DRS performance as Active Aero baseline
        self.active_aero_baseline = self.config.get('active_aero_ratings', {
            'Red Bull Racing': 0.95,
            'Mercedes': 0.92,
            'Ferrari': 0.90,
            'McLaren': 0.88,
            'Aston Martin': 0.85,
            'Alpine': 0.82,
            'Williams': 0.80,
            'RB': 0.78,
            'Kick Sauber': 0.75,
            'Haas': 0.73,
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration, falling back to the built-in 2026 baselines."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(f"Could not load {config_path}: {exc}. Using defaults.")
            return {}
    
    def apply_historical_weight_decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply reduced weighting to historical constructor performance.
        Since 2026 is a reset, recent team dominance is less predictive.
        """
        weight_decay = self.regs_config.get('historical_weight_decay', {
            2025: 0.4,
            2024: 0.3,
            2023: 0.2,
            2022: 0.1,
        })
        
        df = df.copy()
        df['reg_weight'] = df['year'].map(weight_decay).fillna(0.3)
        
        # Apply to constructor form
        if 'constructor_form' in df.columns:
            df['constructor_form_weighted'] = df['constructor_form'] * df['reg_weight']
        
        return df
    
    def get_new_engine_uncertainty(self, 
                                   constructor_id: str, 
                                   race_number: int) -> float:
        """
        Get uncertainty multiplier for new engine manufacturers.
        Higher uncertainty in first 5 races, then normalizes.
        """
        engine = self.team_engine_2026.get(constructor_id.lower(), 'Unknown')
        
        new_engines = self.regs_config.get('new_engine_manufacturers', [])
        
        for new_engine in new_engines:
            if new_engine['name'] == engine:
                uncertainty_races = new_engine.get('uncertainty_races', 5)
                base_multiplier = new_engine.get('uncertainty_multiplier', 1.25)
                
                if race_number <= uncertainty_races:
                    # Linear decay from max to 1.0
                    decay_factor = 1 - (race_number / uncertainty_races)
                    return 1.0 + (base_multiplier - 1.0) * decay_factor
                    
        return 1.0  # No extra uncertainty
    
    def calculate_active_aero_efficiency(self, 
                                         constructor_name: str,
                                         historical_drs_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate Active Aero efficiency rating for 2026.
        
        Based on:
        1. Historical DRS effectiveness
        2. Team's aerodynamic performance in 2022 regulation change
        """
        # Get baseline from config
        baseline = self.active_aero_baseline.get(constructor_name, 0.8)
        
        if historical_drs_data is not None and not historical_drs_data.empty:
            # Calculate from actual DRS data
            team_data = historical_drs_data[
                historical_drs_data['Team'] == constructor_name
            ]
            
            if not team_data.empty:
                # Normalize DRS speed gain
                max_gain = historical_drs_data['DRSSpeedGain'].max()
                if max_gain > 0:
                    team_avg = team_data['DRSSpeedGain'].mean()
                    baseline = (team_avg / max_gain) * 0.3 + baseline * 0.7
        
        return min(1.0, max(0.5, baseline))
    
    def calculate_power_unit_2026_profile(self, 
                                          constructor_id: str,
                                          circuit_type: str) -> Dict[str, float]:
        """
        Calculate Power Unit performance profile for 2026.
        
        2026 PU is 50% Electric / 50% ICE.
        Tracks with long straights require good battery deployment.
        """
        engine = self.team_engine_2026.get(constructor_id.lower(), 'Unknown')

        # Get testing-based reliability score
        testing_reliability = self.testing_data.get_power_unit_reliability_score(engine)

        # Base ERS/Battery performance ratings (adjusted with testing data)
        # Testing showed Mercedes and Ferrari PUs running very reliably
        battery_ratings = {
            'Mercedes': 0.96,      # Excellent testing - 1137 laps
            'Ferrari': 0.92,       # Very strong testing - nearly 1000 laps
            'Honda': 0.90,         # Solid testing for Aston/RB
            'Renault': 0.86,       # Decent testing, some issues
            'Red Bull Powertrains-Ford': 0.82,  # Good for new PU, minor teething
            'Audi': 0.76,          # New manufacturer, calibration issues expected
        }

        base_battery = battery_ratings.get(engine, 0.80) * (0.9 + testing_reliability * 0.1)

        # ICE performance ratings (adjusted with testing data)
        ice_ratings = {
            'Mercedes': 0.94,      # Strong Bahrain times
            'Ferrari': 0.96,       # Fastest in Bahrain testing
            'Honda': 0.92,         # Good pace
            'Renault': 0.87,       # Moderate pace
            'Red Bull Powertrains-Ford': 0.84,  # Decent for first outing
            'Audi': 0.79,          # Slowest but expected for new PU
        }

        base_ice = ice_ratings.get(engine, 0.85) * (0.9 + testing_reliability * 0.1)
        
        # Circuit type affects which component matters more
        if circuit_type == 'power_hungry':
            # Long straights - battery deployment crucial
            battery_weight = 0.6
            ice_weight = 0.4
        elif circuit_type == 'high_downforce':
            # Slow corners - ICE + traction matters more
            battery_weight = 0.4
            ice_weight = 0.6
        else:
            # Balanced
            battery_weight = 0.5
            ice_weight = 0.5
        
        combined_pu_score = base_battery * battery_weight + base_ice * ice_weight
        
        return {
            'battery_score': base_battery,
            'ice_score': base_ice,
            'combined_pu_score': combined_pu_score,
            'circuit_type': circuit_type,
        }
    
    def apply_weight_reduction_effect(self, 
                                      df: pd.DataFrame,
                                      reference_year: int = 2022) -> pd.DataFrame:
        """
        Use 2022 data as a "reverse proxy" for how teams handle major physical changes.
        
        2022 was when cars got heavier and wider.
        2026 cars are -30kg lighter with narrower tires.
        Teams that adapted well in 2022 may have an advantage.
        """
        df = df.copy()
        
        # Calculate each team's performance in 2022 vs 2021 (adaptation score)
        if reference_year in df['year'].unique():
            prev_year = reference_year - 1
            
            if prev_year in df['year'].unique():
                # Calculate team ranking change from 2021 to 2022
                standings_2021 = df[df['year'] == prev_year].groupby('constructor_id')['points'].sum()
                standings_2022 = df[df['year'] == reference_year].groupby('constructor_id')['points'].sum()
                
                # Rank change
                rank_2021 = standings_2021.rank(ascending=False)
                rank_2022 = standings_2022.rank(ascending=False)
                
                adaptation_score = {}
                for team in standings_2022.index:
                    if team in rank_2021.index:
                        # Positive = improved in 2022, Negative = declined
                        rank_change = rank_2021[team] - rank_2022[team]
                        # Normalize to 0-1 scale (0.5 = no change)
                        adaptation_score[team] = 0.5 + (rank_change / 10)
                    else:
                        adaptation_score[team] = 0.5
                
                df['reg_adaptation_score'] = df['constructor_id'].map(adaptation_score).fillna(0.5)
                df['reg_adaptation_score'] = df['reg_adaptation_score'].clip(0, 1)
        else:
            df['reg_adaptation_score'] = 0.5
        
        return df
    
    def get_2026_driver_lineup(self) -> Dict[str, List[str]]:
        """
        Get confirmed 2026 driver lineup.
        Updated with official driver announcements.
        """
        return {
            'Red Bull Racing': ['Max Verstappen', 'Liam Lawson'],
            'Mercedes': ['George Russell', 'Kimi Antonelli'],
            'Ferrari': ['Charles Leclerc', 'Lewis Hamilton'],
            'McLaren': ['Lando Norris', 'Oscar Piastri'],
            'Aston Martin': ['Fernando Alonso', 'Lance Stroll'],
            'Alpine': ['Pierre Gasly', 'Jack Doohan'],
            'Williams': ['Alex Albon', 'Carlos Sainz'],
            'RB': ['Yuki Tsunoda', 'Isack Hadjar'],
            'Audi (Sauber)': ['Nico Hulkenberg', 'Gabriel Bortoleto'],
            'Haas': ['Esteban Ocon', 'Ollie Bearman'],
        }
    
    def get_constructor_display_name(self, constructor_id: str) -> str:
        """Map a constructor id onto the display name used by lookup tables."""
        return constructor_display_name(constructor_id)

    def get_2026_grid_entries(self) -> List[Dict[str, str]]:
        """
        Full 2026 entry list as records.

        Used as the fallback driver set when no historical data is available
        for the season being predicted.
        """
        return [
            {
                'driver_id': driver_id,
                'driver_code': code,
                'driver_name': name,
                'constructor_id': constructor_id,
                'constructor_name': self.get_constructor_display_name(constructor_id),
            }
            for driver_id, code, name, constructor_id in GRID_2026
        ]

    def calculate_2026_team_uncertainty(self,
                                        constructor_id: str,
                                        race_number: int = 1) -> Dict[str, float]:
        """
        Calculate overall uncertainty for a team in 2026.
        
        Combines:
        - New engine uncertainty
        - Regulation adaptation uncertainty
        - Driver change uncertainty
        """
        base_uncertainty = 1.0
        
        # Engine uncertainty
        engine_mult = self.get_new_engine_uncertainty(constructor_id, race_number)
        
        # Teams with new drivers have higher uncertainty
        driver_uncertainty = 1.0
        lineup = self.get_2026_driver_lineup()

        team_name = self.get_constructor_display_name(constructor_id)
        team_drivers = lineup.get(team_name, ['TBD', 'TBD'])
        
        # Count TBD drivers
        tbd_count = sum(1 for d in team_drivers if d == 'TBD')
        driver_uncertainty = 1.0 + (tbd_count * 0.1)
        
        # Early season uncertainty (first 5 races)
        season_uncertainty = 1.0
        if race_number <= 3:
            season_uncertainty = 1.3
        elif race_number <= 5:
            season_uncertainty = 1.15
        elif race_number <= 8:
            season_uncertainty = 1.05

        # Testing-based uncertainty adjustment
        # Teams that tested well have lower uncertainty
        testing_adjustment = self.testing_data.get_team_testing_uncertainty_adjustment(team_name)

        # Combined uncertainty
        total_uncertainty = base_uncertainty * engine_mult * driver_uncertainty * season_uncertainty * testing_adjustment
        
        return {
            'total': total_uncertainty,
            'engine': engine_mult,
            'driver': driver_uncertainty,
            'season': season_uncertainty,
            'testing': testing_adjustment,
        }
    
    def get_testing_performance_adjustment(self, constructor_name: str) -> float:
        """
        Get performance adjustment based on 2026 pre-season testing.

        Returns position adjustment (negative = better, positive = worse)
        Teams that tested well get a small boost.
        """
        testing_rating = self.testing_data.get_testing_performance_rating(constructor_name)

        # Convert rating to position adjustment
        # Rating 1.0 (best) = -0.5 positions
        # Rating 0.5 (average) = 0 positions
        # Rating 0.3 (worst) = +0.4 positions
        adjustment = (0.8 - testing_rating) * 2.0

        return adjustment

    def adjust_predictions_for_2026(self,
                                    predictions: Dict[str, float],
                                    uncertainties: Dict[str, float],
                                    constructor_map: Dict[str, str],
                                    race_number: int = 1,
                                    circuit_type: str = 'normal') -> Dict[str, Dict[str, float]]:
        """
        Adjust ML predictions for 2026 regulation factors.
        
        Args:
            predictions: Dict of driver_id -> predicted position
            uncertainties: Dict of driver_id -> base uncertainty
            constructor_map: Dict of driver_id -> constructor_id
            race_number: Race number in the season
            circuit_type: Type of circuit
            
        Returns:
            Dict with adjusted predictions and uncertainties
        """
        adjusted = {
            'predictions': {},
            'uncertainties': {},
            'adjustments': {},
        }
        
        for driver, base_pred in predictions.items():
            constructor = constructor_map.get(driver, 'unknown')
            base_uncertainty = uncertainties.get(driver, 1.5)
            
            # Get 2026-specific uncertainty
            team_uncertainty = self.calculate_2026_team_uncertainty(constructor, race_number)
            
            # Get power unit profile for this circuit
            pu_profile = self.calculate_power_unit_2026_profile(constructor, circuit_type)
            
            # Adjust prediction based on PU score
            pu_adjustment = (pu_profile['combined_pu_score'] - 0.85) * 2  # Scale around average
            
            # Active aero effect
            team_name = self.get_constructor_display_name(constructor)
            aero_score = self.calculate_active_aero_efficiency(team_name)
            aero_adjustment = (aero_score - 0.85) * 1.5

            # Testing performance adjustment
            testing_adjustment = self.get_testing_performance_adjustment(team_name)

            # Apply adjustments (negative = better position)
            adjusted_pred = base_pred - pu_adjustment - aero_adjustment - testing_adjustment
            adjusted_pred = max(1, min(20, adjusted_pred))
            
            # Apply uncertainty multiplier
            adjusted_uncertainty = base_uncertainty * team_uncertainty['total']
            
            adjusted['predictions'][driver] = adjusted_pred
            adjusted['uncertainties'][driver] = adjusted_uncertainty
            adjusted['adjustments'][driver] = {
                'pu_adjustment': pu_adjustment,
                'aero_adjustment': aero_adjustment,
                'testing_adjustment': testing_adjustment,
                'uncertainty_factor': team_uncertainty,
            }
        
        return adjusted


def apply_2026_adjustments(df: pd.DataFrame, 
                          config_path: str = "config/settings.yaml") -> pd.DataFrame:
    """
    Main function to apply all 2026 regulation adjustments to training data.
    """
    regs = Regulations2026(config_path)
    
    # Apply historical weight decay
    df = regs.apply_historical_weight_decay(df)
    
    # Apply weight reduction effect (2022 adaptation)
    df = regs.apply_weight_reduction_effect(df)
    
    logger.info("Applied 2026 regulation adjustments")
    
    return df


