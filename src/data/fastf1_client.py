"""
F1 Data Pipeline - FastF1 Integration
Fetches telemetry, lap times, weather data, and session information.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FastF1Client:
    """Client for FastF1 telemetry and session data."""
    
    def __init__(self, cache_dir: str = "./cache/fastf1"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        
    def get_session(self, year: int, race: Union[str, int], session_type: str = 'R') -> fastf1.core.Session:
        """
        Load a session.
        session_type: 'R' (Race), 'Q' (Qualifying), 'FP1', 'FP2', 'FP3', 'S' (Sprint)
        """
        session = fastf1.get_session(year, race, session_type)
        session.load()
        return session
    
    def get_lap_data(self, year: int, race: Union[str, int], session_type: str = 'R') -> pd.DataFrame:
        """
        Get detailed lap data for a session.
        Includes lap times, tire compounds, sector times, etc.
        """
        try:
            session = self.get_session(year, race, session_type)
            laps = session.laps.copy()
            
            # Add computed columns
            laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
            laps['Sector1Seconds'] = laps['Sector1Time'].dt.total_seconds()
            laps['Sector2Seconds'] = laps['Sector2Time'].dt.total_seconds()
            laps['Sector3Seconds'] = laps['Sector3Time'].dt.total_seconds()
            
            # Add session info
            laps['Year'] = year
            laps['Race'] = race if isinstance(race, str) else session.event['EventName']
            laps['SessionType'] = session_type
            
            return laps
            
        except Exception as e:
            logger.error(f"Failed to get lap data for {year} {race}: {e}")
            return pd.DataFrame()
    
    def calculate_tire_degradation(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate tire degradation as the slope of lap times during each stint.
        Higher positive slope = worse degradation.
        """
        if laps_df.empty:
            return pd.DataFrame()
            
        degradation_data = []
        
        # Group by driver and stint
        for (driver, stint), stint_laps in laps_df.groupby(['Driver', 'Stint']):
            # Filter valid laps (no pit in/out, no slow laps)
            valid_laps = stint_laps[
                (stint_laps['IsAccurate'] == True) & 
                (stint_laps['LapTimeSeconds'].notna())
            ].copy()
            
            if len(valid_laps) < 3:
                continue
                
            # Calculate degradation slope using linear regression
            valid_laps = valid_laps.sort_values('LapNumber')
            lap_nums = valid_laps['LapNumber'].values
            lap_times = valid_laps['LapTimeSeconds'].values
            
            # Linear regression: time = slope * lap_number + intercept
            if len(lap_nums) >= 3:
                slope, intercept = np.polyfit(lap_nums, lap_times, 1)
                
                # Normalize by average lap time (deg per lap as percentage)
                avg_lap_time = np.mean(lap_times)
                deg_per_lap_pct = (slope / avg_lap_time) * 100
                
                degradation_data.append({
                    'Driver': driver,
                    'Stint': stint,
                    'Compound': valid_laps['Compound'].iloc[0],
                    'StintLength': len(valid_laps),
                    'DegSlopeSeconds': slope,
                    'DegPerLapPct': deg_per_lap_pct,
                    'AvgLapTime': avg_lap_time,
                    'StartLap': lap_nums[0],
                    'EndLap': lap_nums[-1],
                })
                
        return pd.DataFrame(degradation_data)
    
    def get_weather_data(self, year: int, race: Union[str, int], session_type: str = 'R') -> pd.DataFrame:
        """
        Get weather data for a session.
        Includes air temp, track temp, humidity, rainfall, wind.
        """
        try:
            session = self.get_session(year, race, session_type)
            weather = session.weather_data.copy()
            
            # Add session info
            weather['Year'] = year
            weather['Race'] = race if isinstance(race, str) else session.event['EventName']
            weather['SessionType'] = session_type
            
            # Calculate weather summary
            weather_summary = {
                'Year': year,
                'Race': weather['Race'].iloc[0] if not weather.empty else race,
                'AvgAirTemp': weather['AirTemp'].mean(),
                'AvgTrackTemp': weather['TrackTemp'].mean(),
                'MaxTrackTemp': weather['TrackTemp'].max(),
                'MinTrackTemp': weather['TrackTemp'].min(),
                'TempVariation': weather['TrackTemp'].max() - weather['TrackTemp'].min(),
                'AvgHumidity': weather['Humidity'].mean(),
                'Rainfall': weather['Rainfall'].any(),
                'AvgWindSpeed': weather['WindSpeed'].mean() if 'WindSpeed' in weather.columns else None,
            }
            
            return pd.DataFrame([weather_summary])
            
        except Exception as e:
            logger.error(f"Failed to get weather data for {year} {race}: {e}")
            return pd.DataFrame()
    
    def get_telemetry_for_driver(self, year: int, race: Union[str, int], 
                                  driver: str, lap_number: int = None) -> pd.DataFrame:
        """
        Get detailed telemetry for a specific driver.
        Includes speed, throttle, brake, RPM, gear, DRS.
        """
        try:
            session = self.get_session(year, race, 'R')
            driver_laps = session.laps.pick_drivers(driver)
            
            if lap_number:
                lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
            else:
                lap = driver_laps.pick_fastest()
                
            telemetry = lap.get_telemetry()
            return telemetry
            
        except Exception as e:
            logger.error(f"Failed to get telemetry for {driver} at {year} {race}: {e}")
            return pd.DataFrame()
    
    def calculate_drs_efficiency(self, year: int, race: Union[str, int]) -> pd.DataFrame:
        """
        Calculate DRS efficiency for each driver.
        Measures speed gain in DRS zones - proxy for Active Aero in 2026.
        """
        try:
            session = self.get_session(year, race, 'R')
            
            drs_data = []
            for driver in session.laps['Driver'].unique():
                driver_laps = session.laps.pick_drivers(driver)
                
                # Get fastest lap with DRS usage
                valid_laps = driver_laps[driver_laps['IsAccurate'] == True]
                if valid_laps.empty:
                    continue
                    
                fastest = valid_laps.pick_fastest()
                
                try:
                    telemetry = fastest.get_telemetry()
                    
                    # Calculate DRS usage and effectiveness
                    if 'DRS' in telemetry.columns:
                        drs_active = telemetry[telemetry['DRS'] > 0]
                        drs_inactive = telemetry[telemetry['DRS'] == 0]
                        
                        if len(drs_active) > 0 and len(drs_inactive) > 0:
                            avg_speed_with_drs = drs_active['Speed'].mean()
                            avg_speed_without_drs = drs_inactive['Speed'].mean()
                            drs_speed_gain = avg_speed_with_drs - avg_speed_without_drs
                            drs_usage_pct = len(drs_active) / len(telemetry) * 100
                            
                            drs_data.append({
                                'Driver': driver,
                                'Team': fastest['Team'],
                                'DRSSpeedGain': drs_speed_gain,
                                'DRSUsagePct': drs_usage_pct,
                                'AvgSpeedWithDRS': avg_speed_with_drs,
                            })
                except Exception as exc:
                    logger.debug(f"No usable DRS telemetry for {driver}: {exc}")
                    continue

            return pd.DataFrame(drs_data)
            
        except Exception as e:
            logger.error(f"Failed to calculate DRS efficiency for {year} {race}: {e}")
            return pd.DataFrame()
    
    def get_event_schedule(self, year: int) -> pd.DataFrame:
        """Get the event schedule with track information."""
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule
        except Exception as e:
            logger.error(f"Failed to get schedule for {year}: {e}")
            return pd.DataFrame()
    
    def get_overtaking_data(self, year: int, race: Union[str, int]) -> pd.DataFrame:
        """
        Analyze position changes to calculate overtaking difficulty.
        """
        try:
            session = self.get_session(year, race, 'R')
            laps = session.laps.copy()
            
            overtakes = []
            
            # Group by lap and analyze position changes
            for lap_num in laps['LapNumber'].unique():
                lap_data = laps[laps['LapNumber'] == lap_num][['Driver', 'Position', 'LapNumber']]
                prev_lap = laps[laps['LapNumber'] == lap_num - 1][['Driver', 'Position']]
                
                if prev_lap.empty:
                    continue
                    
                # Merge to compare positions
                merged = lap_data.merge(prev_lap, on='Driver', suffixes=('', '_prev'))
                merged['PositionChange'] = merged['Position_prev'] - merged['Position']
                
                # Count overtakes (position improvements that aren't from pit stops)
                gains = merged[merged['PositionChange'] > 0]
                
                overtakes.append({
                    'LapNumber': lap_num,
                    'OvertakesThisLap': len(gains),
                    'TotalPositionGains': gains['PositionChange'].sum(),
                })
                
            return pd.DataFrame(overtakes)
            
        except Exception as e:
            logger.error(f"Failed to calculate overtaking data for {year} {race}: {e}")
            return pd.DataFrame()


def fetch_fastf1_data(years: List[int], cache_dir: str = "./cache/fastf1") -> Dict[str, pd.DataFrame]:
    """
    Main function to fetch FastF1 data for training.
    Returns dictionary of DataFrames.
    """
    client = FastF1Client(cache_dir)
    
    all_data = {
        'lap_data': [],
        'tire_degradation': [],
        'weather': [],
        'drs_efficiency': [],
        'overtaking': [],
    }
    
    for year in years:
        logger.info(f"Fetching FastF1 data for {year}...")
        
        try:
            schedule = client.get_event_schedule(year)
            
            for _, event in schedule.iterrows():
                if event['EventFormat'] == 'testing':
                    continue
                    
                race_name = event['EventName']
                logger.info(f"  Processing {race_name}...")
                
                try:
                    # Lap data
                    laps = client.get_lap_data(year, race_name, 'R')
                    if not laps.empty:
                        all_data['lap_data'].append(laps)
                        
                        # Tire degradation from lap data
                        deg = client.calculate_tire_degradation(laps)
                        if not deg.empty:
                            deg['Year'] = year
                            deg['Race'] = race_name
                            all_data['tire_degradation'].append(deg)
                    
                    # Weather
                    weather = client.get_weather_data(year, race_name, 'R')
                    if not weather.empty:
                        all_data['weather'].append(weather)
                    
                    # DRS efficiency
                    drs = client.calculate_drs_efficiency(year, race_name)
                    if not drs.empty:
                        drs['Year'] = year
                        drs['Race'] = race_name
                        all_data['drs_efficiency'].append(drs)
                    
                    # Overtaking data
                    overtakes = client.get_overtaking_data(year, race_name)
                    if not overtakes.empty:
                        overtakes['Year'] = year
                        overtakes['Race'] = race_name
                        all_data['overtaking'].append(overtakes)
                        
                except Exception as e:
                    logger.warning(f"  Failed to process {race_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to get schedule for {year}: {e}")
            continue
    
    # Combine all data
    combined = {}
    for key, dfs in all_data.items():
        if dfs:
            combined[key] = pd.concat(dfs, ignore_index=True)
        else:
            combined[key] = pd.DataFrame()
            
    logger.info("FastF1 data fetch complete!")
    return combined


