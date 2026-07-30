"""
F1 Data Pipeline - Main Integration Module
Merges Ergast and FastF1 data into a unified training dataset.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import yaml

from ..constants import constructor_display_name
from .ergast_api import fetch_ergast_data, ErgastAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataPipeline:
    """
    Main data pipeline that integrates Ergast and FastF1 data.
    Creates a unified DataFrame where each row is a Driver-Race-Year combination.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.ergast_api = ErgastAPI(
            base_url=self.config.get('data', {}).get('ergast_base_url')
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def _resolve_years(self, years: Optional[List[int]]) -> List[int]:
        if years:
            return list(years)
        return self.config.get('data', {}).get('years_to_fetch', [2023, 2024])

    def _use_fastf1(self, use_fastf1: Optional[bool]) -> bool:
        if use_fastf1 is not None:
            return use_fastf1
        return bool(self.config.get('data', {}).get('use_fastf1', True))

    def fetch_all_data(self,
                       years: List[int] = None,
                       use_fastf1: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data from Ergast and (optionally) FastF1.

        FastF1 downloads full session telemetry and takes tens of minutes per
        season, so it can be skipped - the pipeline degrades to the Ergast-only
        feature set with config-based defaults for the telemetry features.
        """
        years = self._resolve_years(years)
        logger.info(f"Fetching data for years: {years}")

        cache_dir = self.config.get('data', {}).get('cache_dir', './cache')
        ergast_data = fetch_ergast_data(years, cache_dir=cache_dir, api=self.ergast_api)

        if self._use_fastf1(use_fastf1):
            # Imported lazily: fastf1 is heavy and only needed on this path.
            from .fastf1_client import fetch_fastf1_data

            fastf1_data = fetch_fastf1_data(years, cache_dir=f"{cache_dir.rstrip('/')}/fastf1")
        else:
            logger.info("Skipping FastF1 telemetry (data.use_fastf1 is disabled)")
            fastf1_data = {}

        return {
            'ergast': ergast_data,
            'fastf1': fastf1_data,
        }

    def build_training_dataset(self,
                               years: List[int] = None,
                               use_fastf1: Optional[bool] = None) -> pd.DataFrame:
        """
        Build the unified training dataset.
        Each row represents a Driver-Race-Year combination with all features.
        """
        years = self._resolve_years(years)

        # Fetch raw data
        raw_data = self.fetch_all_data(years, use_fastf1=use_fastf1)
        ergast = raw_data['ergast']
        fastf1 = raw_data['fastf1']

        if ergast.get('race_results') is None or ergast['race_results'].empty:
            raise RuntimeError(
                f"No race results returned for {years}. Check network access to "
                f"{self.ergast_api.BASE_URL} or pick different years."
            )

        # Start with race results as the base
        df = ergast['race_results'].copy()
        
        # Add qualifying data
        df = self._merge_qualifying(df, ergast['qualifying'])
        
        # Add pit stop statistics
        df = self._merge_pit_stops(df, ergast['pit_stops'])
        
        # Add reliability scores
        df = self._merge_reliability(df, ergast['reliability'])
        
        # Add tire degradation
        df = self._merge_tire_degradation(df, fastf1.get('tire_degradation', pd.DataFrame()))
        
        # Add weather impact
        df = self._merge_weather(df, fastf1.get('weather', pd.DataFrame()))
        
        # Add DRS efficiency (proxy for Active Aero)
        df = self._merge_drs_efficiency(df, fastf1.get('drs_efficiency', pd.DataFrame()))
        
        # Add track categories
        df = self._add_track_categories(df)
        
        # Calculate position delta (Grid - Finish)
        df['position_delta'] = df['grid_position'] - df['finish_position'].fillna(20)
        
        # Add historical weighting for 2026
        df = self._apply_historical_weights(df)
        
        logger.info(f"Built training dataset with {len(df)} rows and {len(df.columns)} features")
        
        return df
    
    def _merge_qualifying(self, df: pd.DataFrame, quali_df: pd.DataFrame) -> pd.DataFrame:
        """Merge qualifying data."""
        if quali_df.empty:
            return df
            
        quali_agg = quali_df.groupby(['year', 'round', 'driver_id']).first().reset_index()
        quali_agg = quali_agg[['year', 'round', 'driver_id', 'quali_position']]
        
        df = df.merge(quali_agg, on=['year', 'round', 'driver_id'], how='left')
        return df
    
    def _merge_pit_stops(self, df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
        """Merge pit stop statistics."""
        if pit_df.empty:
            df['avg_pit_time'] = np.nan
            df['num_pit_stops'] = np.nan
            return df
        
        # Calculate per-race pit statistics for each driver
        pit_stats = pit_df.groupby(['year', 'round', 'driver_id']).agg({
            'duration_seconds': ['mean', 'min', 'count']
        }).reset_index()
        pit_stats.columns = ['year', 'round', 'driver_id', 'avg_pit_time', 'best_pit_time', 'num_pit_stops']
        
        # Also calculate team averages
        team_pit = df.merge(pit_df, on=['year', 'round', 'driver_id'], how='left')
        team_pit_stats = team_pit.groupby(['year', 'constructor_id']).agg({
            'duration_seconds': 'mean'
        }).reset_index()
        team_pit_stats.columns = ['year', 'constructor_id', 'team_avg_pit_time']
        
        df = df.merge(pit_stats, on=['year', 'round', 'driver_id'], how='left')
        df = df.merge(team_pit_stats, on=['year', 'constructor_id'], how='left')
        
        return df
    
    def _merge_reliability(self, df: pd.DataFrame, reliability_df: pd.DataFrame) -> pd.DataFrame:
        """Merge reliability scores."""
        if reliability_df.empty:
            df['reliability_score'] = 0.9  # Default
            return df
            
        reliability_df = reliability_df[['constructor_id', 'reliability_score']]
        df = df.merge(reliability_df, on='constructor_id', how='left')
        df['reliability_score'] = df['reliability_score'].fillna(0.9)
        
        return df
    
    def _merge_tire_degradation(self, df: pd.DataFrame, deg_df: pd.DataFrame) -> pd.DataFrame:
        """Merge tire degradation data."""
        if deg_df.empty:
            df['avg_deg_per_lap_pct'] = np.nan
            df['avg_deg_slope'] = np.nan
            df['season_avg_deg'] = np.nan
            return df
        
        # Calculate average degradation per driver per race
        deg_agg = deg_df.groupby(['Year', 'Race', 'Driver']).agg({
            'DegPerLapPct': 'mean',
            'DegSlopeSeconds': 'mean',
        }).reset_index()
        deg_agg.columns = ['year', 'race_name', 'driver_code', 'avg_deg_per_lap_pct', 'avg_deg_slope']
        
        # Also get team averages
        team_deg = deg_df.groupby(['Year']).agg({
            'DegPerLapPct': 'mean'
        }).reset_index()
        team_deg.columns = ['year', 'season_avg_deg']
        
        df = df.merge(deg_agg, 
                     left_on=['year', 'race_name', 'driver_code'],
                     right_on=['year', 'race_name', 'driver_code'],
                     how='left')
        df = df.merge(team_deg, on='year', how='left')
        
        return df
    
    def _merge_weather(self, df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data."""
        if weather_df.empty:
            df['avg_track_temp'] = np.nan
            df['is_wet_race'] = False
            return df
        
        weather_cols = ['Year', 'Race', 'AvgTrackTemp', 'TempVariation', 'Rainfall']
        weather_df = weather_df[weather_cols].copy()
        weather_df.columns = ['year', 'race_name', 'avg_track_temp', 'temp_variation', 'is_wet_race']
        
        df = df.merge(weather_df, on=['year', 'race_name'], how='left')
        df['is_wet_race'] = df['is_wet_race'].fillna(False)
        
        return df
    
    def _merge_drs_efficiency(self, df: pd.DataFrame, drs_df: pd.DataFrame) -> pd.DataFrame:
        """Merge DRS efficiency as proxy for Active Aero capability."""
        aero_ratings = self.config.get('active_aero_ratings', {})

        if drs_df.empty:
            # No telemetry: fall back to the configured Active Aero ratings,
            # matched on the canonical team display name.
            display_names = df['constructor_id'].map(constructor_display_name)
            df['active_aero_efficiency'] = (
                display_names.map(aero_ratings)
                .fillna(df['constructor_name'].map(aero_ratings))
                .fillna(0.8)
            )
            return df


        # Calculate team average DRS efficiency
        drs_team = drs_df.groupby(['Year', 'Team']).agg({
            'DRSSpeedGain': 'mean',
            'DRSUsagePct': 'mean',
        }).reset_index()
        drs_team.columns = ['year', 'constructor_name', 'drs_speed_gain', 'drs_usage_pct']
        
        df = df.merge(drs_team, on=['year', 'constructor_name'], how='left')
        
        # Normalize to 0-1 scale for Active Aero efficiency
        if 'drs_speed_gain' in df.columns:
            max_gain = df['drs_speed_gain'].max()
            if max_gain > 0:
                df['active_aero_efficiency'] = df['drs_speed_gain'] / max_gain
            else:
                df['active_aero_efficiency'] = 0.8
        else:
            df['active_aero_efficiency'] = 0.8
            
        return df
    
    def _add_track_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track category features."""
        track_cats = self.config.get('track_categories', {})
        
        # Create binary columns for each category
        for category, tracks in track_cats.items():
            df[f'is_{category}'] = df['circuit_name'].apply(
                lambda x: 1 if any(t.lower() in x.lower() for t in tracks) else 0
            )
        
        return df
    
    def _apply_historical_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply historical weights based on 2026 regulation reset logic.
        More recent years get less weight since 2026 is a reset.
        """
        weight_decay = self.config.get('regulations_2026', {}).get('historical_weight_decay', {
            2025: 0.4,
            2024: 0.3,
            2023: 0.2,
            2022: 0.1,
        })
        
        df['historical_weight'] = df['year'].map(weight_decay).fillna(0.5)
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str = "data/training_data.parquet"):
        """Save the training dataset."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output, index=False)
        logger.info(f"Saved training data to {output_path}")
        
        # Also save as CSV for inspection
        csv_path = output.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV version to {csv_path}")
    
    def load_dataset(self, input_path: str = "data/training_data.parquet") -> pd.DataFrame:
        """Load a previously saved training dataset."""
        return pd.read_parquet(input_path)


def build_pipeline(years: List[int] = None, save: bool = True) -> pd.DataFrame:
    """
    Main function to build the complete data pipeline.
    """
    pipeline = F1DataPipeline()
    
    if years is None:
        years = [2022, 2023, 2024, 2025]
    
    df = pipeline.build_training_dataset(years)
    
    if save:
        pipeline.save_dataset(df)
    
    return df


