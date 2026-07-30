"""
F1 Feature Engineering Module
Transforms raw data into predictive features for the ML model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for F1 race prediction.
    Transforms raw data into meaningful predictive features.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        """
        df = df.copy()
        
        # Driver form (rolling average of last N races)
        df = self._calculate_driver_form(df, window=5)
        
        # Constructor form
        df = self._calculate_constructor_form(df, window=5)
        
        # Grid position features
        df = self._engineer_grid_features(df)
        
        # Track-specific features
        df = self._engineer_track_features(df)
        
        # Weather interaction features
        df = self._engineer_weather_features(df)
        
        # Pit stop features
        df = self._engineer_pit_features(df)
        
        # Power unit profile for 2026
        df = self._engineer_power_unit_features(df)
        
        # Head-to-head features
        df = self._calculate_head_to_head(df)
        
        # Encode categorical variables
        df = self._encode_categoricals(df)
        
        return df
    
    def _calculate_driver_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate rolling average performance for each driver.
        Uses finish position normalized (1=win, 0=last).
        """
        # Sort by date
        df = df.sort_values(['driver_id', 'year', 'round'])
        
        # Calculate position score (inverse normalized)
        df['position_score'] = 1 - (df['finish_position'].fillna(20) - 1) / 19
        
        # Rolling average (excluding current race)
        df['driver_form'] = df.groupby('driver_id')['position_score'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        
        # Fill NaN with average (new drivers)
        df['driver_form'] = df['driver_form'].fillna(0.5)
        
        # Calculate form trend (improving or declining)
        df['form_trend'] = df.groupby('driver_id')['position_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        ) - df.groupby('driver_id')['position_score'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        df['form_trend'] = df['form_trend'].fillna(0)
        
        # Calculate consistency (std dev of recent finishes)
        df['driver_consistency'] = df.groupby('driver_id')['position_score'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
        )
        df['driver_consistency'] = df['driver_consistency'].fillna(0.2)
        
        return df
    
    def _calculate_constructor_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """Calculate rolling constructor performance."""
        # Average points per race for constructor
        constructor_points = df.groupby(['constructor_id', 'year', 'round']).agg({
            'points': 'sum',
            'finish_position': 'mean'
        }).reset_index()
        constructor_points.columns = ['constructor_id', 'year', 'round', 'team_race_points', 'team_avg_finish']
        
        constructor_points = constructor_points.sort_values(['constructor_id', 'year', 'round'])
        
        # Rolling team form
        constructor_points['constructor_form'] = constructor_points.groupby('constructor_id')['team_race_points'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
        )
        constructor_points['constructor_form'] = constructor_points['constructor_form'].fillna(10)
        
        df = df.merge(constructor_points[['constructor_id', 'year', 'round', 'constructor_form']], 
                     on=['constructor_id', 'year', 'round'], how='left')
        
        return df
    
    def _engineer_grid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from grid/qualifying position."""
        # Grid position is already the #1 predictor
        df['grid_position'] = df['grid_position'].fillna(20)
        
        # Front row indicator
        df['is_front_row'] = (df['grid_position'] <= 2).astype(int)
        
        # Top 5 indicator
        df['is_top5_grid'] = (df['grid_position'] <= 5).astype(int)
        
        # Top 10 indicator
        df['is_top10_grid'] = (df['grid_position'] <= 10).astype(int)
        
        # Did the driver gain places relative to the grid in this race?
        gained = (df['grid_position'] - df['finish_position'].fillna(20)) > 0
        df['_gained_places'] = gained.astype(float)

        # Historical grid-to-finish conversion rate.
        # Expanding mean shifted by one race so the current result never leaks
        # into its own feature row.
        df['grid_conversion_rate'] = (
            df.groupby('driver_id')['_gained_places']
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(0.5)
        )

        # Track-specific grid conversion, same leak-free construction.
        df['track_grid_conversion'] = (
            df.groupby(['driver_id', 'circuit_id'])['_gained_places']
            .transform(lambda x: x.expanding().mean().shift(1))
            .fillna(df['grid_conversion_rate'])
        )

        df = df.drop(columns=['_gained_places'])

        return df
    
    def _engineer_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer track-specific features."""
        if 'position_delta' not in df.columns:
            # Normally supplied by the data pipeline; derive it if absent.
            df['position_delta'] = df['grid_position'] - df['finish_position'].fillna(20)

        # Track experience (number of races at this circuit)
        track_exp = df.groupby(['driver_id', 'circuit_id']).cumcount()
        df['track_experience'] = track_exp
        
        # Historical track performance
        track_perf = df.groupby(['driver_id', 'circuit_id'])['position_score'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['track_historical_perf'] = track_perf.fillna(0.5)
        
        # Overtaking difficulty index (from historical data)
        # Higher number = harder to overtake.  This is a circuit property rather
        # than a driver outcome, so a full-sample average is acceptable here.
        overtake_idx = (
            df.groupby('circuit_id')['position_delta']
            .apply(lambda x: 1 - (x.abs().mean() / 10))
            .reset_index()
        )
        overtake_idx.columns = ['circuit_id', 'overtake_difficulty']

        df = df.merge(overtake_idx, on='circuit_id', how='left')
        df['overtake_difficulty'] = df['overtake_difficulty'].fillna(0.5).clip(0, 1)
        
        return df
    
    def _engineer_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer weather-related features."""
        # Get rain performance ratings from config
        rain_ratings = self.config.get('rain_performance', {})
        default_rain = rain_ratings.get('default', 0.75)
        
        # Map driver rain performance
        df['rain_skill'] = df['driver_code'].map(rain_ratings).fillna(default_rain)
        
        # Wet race performance boost/penalty
        if 'is_wet_race' in df.columns:
            df['weather_adjusted_skill'] = np.where(
                df['is_wet_race'],
                df['driver_form'] * df['rain_skill'],
                df['driver_form']
            )
        else:
            df['weather_adjusted_skill'] = df['driver_form']
        
        # Temperature effect on tire degradation
        if 'avg_track_temp' in df.columns:
            # Higher track temp = more degradation potential
            df['temp_deg_factor'] = (df['avg_track_temp'] - 30) / 20  # Normalized around 30°C
            df['temp_deg_factor'] = df['temp_deg_factor'].fillna(0).clip(-1, 1)
        else:
            df['temp_deg_factor'] = 0
        
        return df
    
    def _engineer_pit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer pit stop efficiency features."""
        # Team pit efficiency relative to season average
        if 'avg_pit_time' in df.columns:
            season_avg = df.groupby('year')['avg_pit_time'].transform('mean')
            df['pit_efficiency'] = season_avg / df['avg_pit_time'].replace(0, np.nan)
            df['pit_efficiency'] = df['pit_efficiency'].fillna(1.0).clip(0.8, 1.2)
        else:
            df['pit_efficiency'] = 1.0
        
        # Team pit consistency (lower std = more consistent)
        if 'team_avg_pit_time' in df.columns:
            team_pit_std = df.groupby('constructor_id')['avg_pit_time'].transform('std')
            df['pit_consistency'] = 1 - (team_pit_std / team_pit_std.max()).fillna(0)
        else:
            df['pit_consistency'] = 0.8
        
        return df
    
    def _engineer_power_unit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer power unit features for 2026's 50/50 electric/ICE split.
        Maps power-hungry tracks to teams that may struggle.
        """
        # Define engine manufacturers
        engine_map = {
            'red_bull': 'Red Bull Powertrains-Ford',
            'mercedes': 'Mercedes',
            'ferrari': 'Ferrari',
            'mclaren': 'Mercedes',
            'aston_martin': 'Honda',
            'alpine': 'Renault',
            'williams': 'Mercedes',
            'rb': 'Honda',
            'sauber': 'Audi',
            'haas': 'Ferrari',
        }
        
        df['engine_manufacturer'] = df['constructor_id'].map(engine_map).fillna('Unknown')
        
        # New engine risk for 2026
        new_engines = self.config.get('regulations_2026', {}).get('new_engine_manufacturers', [])
        new_engine_names = [e['name'] for e in new_engines]
        
        df['is_new_engine'] = df['engine_manufacturer'].isin(new_engine_names).astype(int)
        
        # Power track indicator (from config)
        power_tracks = self.config.get('track_categories', {}).get('power_hungry', [])
        df['is_power_track'] = df['circuit_name'].apply(
            lambda x: 1 if any(t.lower() in x.lower() for t in power_tracks) else 0
        )
        
        # Interaction: new engine on power track = higher risk
        df['power_track_risk'] = df['is_new_engine'] * df['is_power_track']
        
        return df
    
    def _calculate_head_to_head(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate teammate head-to-head statistics."""
        # Get teammate pairs
        teammates = df.groupby(['year', 'round', 'constructor_id'])['driver_id'].apply(list).reset_index()
        teammates = teammates[teammates['driver_id'].apply(len) == 2]
        
        h2h_stats = []
        for _, row in teammates.iterrows():
            d1, d2 = row['driver_id']
            year, rnd = row['year'], row['round']
            
            race_data = df[(df['year'] == year) & (df['round'] == rnd)]
            d1_pos = race_data[race_data['driver_id'] == d1]['finish_position'].values
            d2_pos = race_data[race_data['driver_id'] == d2]['finish_position'].values
            
            if len(d1_pos) > 0 and len(d2_pos) > 0:
                d1_pos, d2_pos = d1_pos[0], d2_pos[0]
                h2h_stats.append({'year': year, 'round': rnd, 'driver_id': d1, 
                                 'beat_teammate': 1 if d1_pos < d2_pos else 0})
                h2h_stats.append({'year': year, 'round': rnd, 'driver_id': d2, 
                                 'beat_teammate': 1 if d2_pos < d1_pos else 0})
        
        if h2h_stats:
            h2h_df = pd.DataFrame(h2h_stats)
            
            # Calculate rolling teammate battle rate
            h2h_df = h2h_df.sort_values(['driver_id', 'year', 'round'])
            h2h_df['teammate_battle_rate'] = h2h_df.groupby('driver_id')['beat_teammate'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
            )
            
            df = df.merge(h2h_df[['year', 'round', 'driver_id', 'teammate_battle_rate']],
                         on=['year', 'round', 'driver_id'], how='left')

        if 'teammate_battle_rate' in df.columns:
            df['teammate_battle_rate'] = df['teammate_battle_rate'].fillna(0.5)
        else:
            # No complete teammate pairs in this dataset - fall back to neutral.
            df['teammate_battle_rate'] = 0.5

        return df
    
    CATEGORICAL_COLUMNS = ['driver_id', 'constructor_id', 'circuit_id', 'engine_manufacturer']

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit label encoders on the training data and encode categoricals."""
        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        return df

    def transform_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply already-fitted label encoders to new (prediction-time) rows.

        Labels never seen during training - a 2026 rookie, a renamed team, a new
        circuit - are encoded as -1 rather than raising, so a prediction can
        still be produced for them.
        """
        df = df.copy()

        for col in self.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue

            encoder = self.label_encoders.get(col)
            if encoder is None:
                df[f'{col}_encoded'] = -1
                continue

            known = {label: idx for idx, label in enumerate(encoder.classes_)}
            df[f'{col}_encoded'] = df[col].astype(str).map(known).fillna(-1).astype(int)

        return df


    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns for the model."""
        return [
            # Core features
            'grid_position',
            'driver_form',
            'form_trend',
            'driver_consistency',
            'constructor_form',
            
            # Grid features
            'is_front_row',
            'is_top5_grid',
            'is_top10_grid',
            'grid_conversion_rate',
            'track_grid_conversion',
            
            # Track features
            'track_experience',
            'track_historical_perf',
            'overtake_difficulty',
            
            # Track categories
            'is_high_speed',
            'is_high_downforce',
            'is_street_circuit',
            'is_power_hungry',
            
            # Weather features
            'rain_skill',
            'weather_adjusted_skill',
            'temp_deg_factor',
            
            # Pit features
            'pit_efficiency',
            'pit_consistency',
            
            # Reliability
            'reliability_score',
            
            # Tire degradation
            'avg_deg_per_lap_pct',
            'season_avg_deg',
            
            # Power unit / 2026 features
            'is_new_engine',
            'is_power_track',
            'power_track_risk',
            'active_aero_efficiency',
            
            # Head-to-head
            'teammate_battle_rate',
            
            # Historical weight
            'historical_weight',
            
            # Encoded categoricals
            'driver_id_encoded',
            'constructor_id_encoded',
            'circuit_id_encoded',
        ]
    
    def get_target_column(self) -> str:
        """Get the target column name."""
        return 'finish_position'


def engineer_features(df: pd.DataFrame, config: Dict = None) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Main function to engineer features.
    Returns the engineered DataFrame and the FeatureEngineer instance.
    """
    engineer = FeatureEngineer(config)
    df_engineered = engineer.engineer_features(df)
    
    logger.info(f"Engineered {len(engineer.get_feature_columns())} features")
    
    return df_engineered, engineer


