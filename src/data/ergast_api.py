"""
F1 Data Pipeline - Ergast API Integration
Fetches historical F1 data including results, pit stops, and reliability stats.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import os
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErgastAPI:
    """Client for the Ergast F1 API (using Jolpica-F1 mirror since Ergast shut down in 2024)."""

    DEFAULT_BASE_URL = "https://api.jolpi.ca/ergast/f1"
    # Public mirror; overridable per-instance or via the F1_ERGAST_BASE_URL env var.
    BASE_URL = DEFAULT_BASE_URL

    def __init__(self,
                 cache_enabled: bool = True,
                 base_url: Optional[str] = None,
                 request_delay: float = 3.0):
        self.cache_enabled = cache_enabled
        self.base_url = (base_url
                         or os.environ.get('F1_ERGAST_BASE_URL')
                         or self.DEFAULT_BASE_URL).rstrip('/')
        self.request_delay = request_delay
        self._cache: Dict = {}

    # The Jolpica mirror silently clamps `limit` to 100, so every collection
    # endpoint has to be paged rather than pulled in one request.
    PAGE_SIZE = 100
    MAX_PAGES = 60

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
    def _make_request(self, endpoint: str, limit: int = None, offset: int = 0) -> Dict:
        """Fetch a single page with retry logic and rate limiting."""
        limit = self.PAGE_SIZE if limit is None else limit
        url = f"{self.base_url}/{endpoint}.json?limit={limit}&offset={offset}"

        if self.cache_enabled and url in self._cache:
            return self._cache[url]

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        if self.cache_enabled:
            self._cache[url] = data

        # Rate limiting - Jolpica-F1 returns 429 without a delay between calls
        if self.request_delay:
            time.sleep(self.request_delay)
        return data

    def _paginate(self, endpoint: str):
        """Yield every MRData page for an endpoint until `total` is exhausted."""
        offset = 0
        for _ in range(self.MAX_PAGES):
            data = self._make_request(endpoint, limit=self.PAGE_SIZE, offset=offset)
            yield data

            mrdata = data.get('MRData', {})
            total = int(mrdata.get('total', 0))
            offset += self.PAGE_SIZE
            if offset >= total:
                return

        logger.warning(f"Stopped paging {endpoint} after {self.MAX_PAGES} pages")

    def _fetch_races(self, endpoint: str, results_key: Optional[str] = None) -> List[Dict]:
        """
        Fetch every race object for an endpoint, stitching paged results back
        together. A single race's results can straddle a page boundary, so
        consecutive entries for the same season/round are merged.
        """
        races: List[Dict] = []

        for page in self._paginate(endpoint):
            for race in page.get('MRData', {}).get('RaceTable', {}).get('Races', []):
                same_race = (
                    races
                    and races[-1].get('season') == race.get('season')
                    and races[-1].get('round') == race.get('round')
                )
                if same_race and results_key:
                    races[-1].setdefault(results_key, []).extend(race.get(results_key, []))
                elif not same_race:
                    races.append(race)

        return races

    def _fetch_standings(self, endpoint: str, standings_key: str) -> List[Dict]:
        """Fetch a full standings table, stitching pages back together."""
        entries: List[Dict] = []

        for page in self._paginate(endpoint):
            lists = page.get('MRData', {}).get('StandingsTable', {}).get('StandingsLists', [])
            for standings_list in lists:
                entries.extend(standings_list.get(standings_key, []))

        return entries


    def get_race_results(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch race results for a given year.
        Returns DataFrame with driver, constructor, grid position, finish position, status.
        """
        if round_num:
            endpoint = f"{year}/{round_num}/results"
        else:
            endpoint = f"{year}/results"

        races = self._fetch_races(endpoint, 'Results')

        results = []
        for race in races:
            race_info = {
                'year': int(race['season']),
                'round': int(race['round']),
                'race_name': race['raceName'],
                'circuit_id': race['Circuit']['circuitId'],
                'circuit_name': race['Circuit']['circuitName'],
                'date': race['date']
            }
            
            for result in race.get('Results', []):
                row = {
                    **race_info,
                    'driver_id': result['Driver']['driverId'],
                    'driver_code': result['Driver'].get('code', result['Driver']['driverId'][:3].upper()),
                    'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    'constructor_id': result['Constructor']['constructorId'],
                    'constructor_name': result['Constructor']['name'],
                    'grid_position': int(result['grid']),
                    'finish_position': int(result['position']) if result['position'].isdigit() else None,
                    'points': float(result['points']),
                    'status': result['status'],
                    'laps_completed': int(result['laps']),
                    'time_millis': int(result['Time']['millis']) if 'Time' in result else None,
                    'fastest_lap_rank': int(result['FastestLap']['rank']) if 'FastestLap' in result else None,
                }
                results.append(row)
                
        return pd.DataFrame(results)
    
    def get_qualifying_results(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Fetch qualifying results for grid position analysis."""
        if round_num:
            endpoint = f"{year}/{round_num}/qualifying"
        else:
            endpoint = f"{year}/qualifying"

        races = self._fetch_races(endpoint, 'QualifyingResults')

        results = []
        for race in races:
            race_info = {
                'year': int(race['season']),
                'round': int(race['round']),
                'circuit_id': race['Circuit']['circuitId'],
            }
            
            for quali in race.get('QualifyingResults', []):
                row = {
                    **race_info,
                    'driver_id': quali['Driver']['driverId'],
                    'constructor_id': quali['Constructor']['constructorId'],
                    'quali_position': int(quali['position']),
                    'q1_time': quali.get('Q1'),
                    'q2_time': quali.get('Q2'),
                    'q3_time': quali.get('Q3'),
                }
                results.append(row)
                
        return pd.DataFrame(results)
    
    def get_pit_stops(self, year: int, round_num: int) -> pd.DataFrame:
        """Fetch pit stop data for a specific race."""
        endpoint = f"{year}/{round_num}/pitstops"
        races = self._fetch_races(endpoint, 'PitStops')

        if not races:
            return pd.DataFrame()


        race = races[0]
        pit_stops = race.get('PitStops', [])
        
        results = []
        for stop in pit_stops:
            results.append({
                'year': int(race['season']),
                'round': int(race['round']),
                'circuit_id': race['Circuit']['circuitId'],
                'driver_id': stop['driverId'],
                'stop_number': int(stop['stop']),
                'lap': int(stop['lap']),
                'duration_seconds': float(stop['duration'].replace(':', '')) if ':' not in stop['duration'] else self._parse_pit_duration(stop['duration']),
            })
            
        return pd.DataFrame(results)
    
    def _parse_pit_duration(self, duration_str: str) -> float:
        """Parse pit stop duration string to seconds."""
        try:
            if ':' in duration_str:
                parts = duration_str.split(':')
                return float(parts[0]) * 60 + float(parts[1])
            return float(duration_str)
        except (TypeError, ValueError):
            return np.nan
    
    def get_all_pit_stops(self, year: int) -> pd.DataFrame:
        """Fetch pit stops for all races in a season."""
        # First get number of races
        schedule = self.get_race_schedule(year)
        num_races = len(schedule)
        
        all_stops = []
        for round_num in range(1, num_races + 1):
            logger.info(f"Fetching pit stops for {year} Round {round_num}")
            try:
                stops = self.get_pit_stops(year, round_num)
                all_stops.append(stops)
            except Exception as e:
                logger.warning(f"Failed to fetch pit stops for {year} R{round_num}: {e}")
                
        if all_stops:
            return pd.concat(all_stops, ignore_index=True)
        return pd.DataFrame()
    
    def get_driver_standings(self, year: int) -> pd.DataFrame:
        """Get driver championship standings."""
        standings = self._fetch_standings(f"{year}/driverStandings", 'DriverStandings')
        if not standings:
            return pd.DataFrame()

        results = []
        for standing in standings:
            results.append({
                'year': year,
                'position': int(standing['position']),
                'driver_id': standing['Driver']['driverId'],
                'driver_name': f"{standing['Driver']['givenName']} {standing['Driver']['familyName']}",
                'constructor_id': standing['Constructors'][0]['constructorId'],
                'points': float(standing['points']),
                'wins': int(standing['wins']),
            })
            
        return pd.DataFrame(results)
    
    def get_constructor_standings(self, year: int) -> pd.DataFrame:
        """Get constructor championship standings."""
        standings = self._fetch_standings(f"{year}/constructorStandings", 'ConstructorStandings')
        if not standings:
            return pd.DataFrame()

        results = []
        for standing in standings:
            results.append({
                'year': year,
                'position': int(standing['position']),
                'constructor_id': standing['Constructor']['constructorId'],
                'constructor_name': standing['Constructor']['name'],
                'points': float(standing['points']),
                'wins': int(standing['wins']),
            })
            
        return pd.DataFrame(results)
    
    def get_race_schedule(self, year: int) -> pd.DataFrame:
        """Get the race calendar for a year."""
        races = self._fetch_races(f"{year}")

        results = []
        for race in races:
            results.append({
                'year': int(race['season']),
                'round': int(race['round']),
                'race_name': race['raceName'],
                'circuit_id': race['Circuit']['circuitId'],
                'circuit_name': race['Circuit']['circuitName'],
                'country': race['Circuit']['Location']['country'],
                'locality': race['Circuit']['Location']['locality'],
                'date': race['date'],
                'time': race.get('time', ''),
            })
            
        return pd.DataFrame(results)
    
    def get_status_data(self, year: int) -> pd.DataFrame:
        """
        Get race status data for reliability analysis.
        Used to calculate DNF rates per engine manufacturer.
        """
        status_list = []
        for page in self._paginate(f"{year}/status"):
            status_list.extend(page.get('MRData', {}).get('StatusTable', {}).get('Status', []))

        results = []
        for status in status_list:
            results.append({
                'year': year,
                'status_id': status['statusId'],
                'status': status['status'],
                'count': int(status['count']),
            })
            
        return pd.DataFrame(results)
    
    def calculate_reliability_scores(self, years: List[int]) -> pd.DataFrame:
        """
        Calculate reliability scores per constructor based on DNF rates.
        """
        all_results = []
        for year in years:
            results = self.get_race_results(year)
            all_results.append(results)
            
        df = pd.concat(all_results, ignore_index=True)
        
        # Define DNF statuses
        dnf_statuses = [
            'Engine', 'Gearbox', 'Transmission', 'Hydraulics', 'Electrical',
            'Mechanical', 'Collision', 'Accident', 'Spun off', 'Wheel', 
            'Suspension', 'Brakes', 'Power Unit', 'ERS', 'Battery', 'Turbo',
            'Oil pressure', 'Water pressure', 'Fuel pressure', 'Overheating'
        ]
        
        # Calculate DNF rate per constructor
        df['is_dnf'] = df['status'].apply(lambda x: 1 if any(s.lower() in x.lower() for s in dnf_statuses) else 0)
        
        reliability = df.groupby('constructor_id').agg({
            'is_dnf': 'sum',
            'driver_id': 'count'  # Total race entries
        }).rename(columns={'driver_id': 'total_entries'})
        
        reliability['dnf_rate'] = reliability['is_dnf'] / reliability['total_entries']
        reliability['reliability_score'] = 1 - reliability['dnf_rate']
        
        return reliability.reset_index()


def fetch_ergast_data(years: List[int],
                      cache_dir: str = "./cache",
                      api: Optional[ErgastAPI] = None) -> Dict[str, pd.DataFrame]:
    """
    Main function to fetch all Ergast data for training.
    Returns dictionary of DataFrames with disk caching support.
    """
    # Create cache directory if it doesn't exist
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"ergast_data_{'_'.join(map(str, years))}.pkl")

    # Check if cached data exists
    if os.path.exists(cache_file):
        logger.info(f"Loading cached Ergast data from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info("Cached data loaded successfully!")
            return cached_data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Fetching fresh data...")

    # Fetch fresh data
    api = api or ErgastAPI(cache_enabled=True)

    all_data = {
        'race_results': [],
        'qualifying': [],
        'pit_stops': [],
        'driver_standings': [],
        'constructor_standings': [],
        'schedules': [],
    }

    for year in years:
        logger.info(f"Fetching Ergast data for {year}...")

        # Race results
        results = api.get_race_results(year)
        all_data['race_results'].append(results)

        # Qualifying
        quali = api.get_qualifying_results(year)
        all_data['qualifying'].append(quali)

        # Pit stops
        pit_stops = api.get_all_pit_stops(year)
        all_data['pit_stops'].append(pit_stops)

        # Standings
        driver_standings = api.get_driver_standings(year)
        all_data['driver_standings'].append(driver_standings)

        constructor_standings = api.get_constructor_standings(year)
        all_data['constructor_standings'].append(constructor_standings)

        # Schedule
        schedule = api.get_race_schedule(year)
        all_data['schedules'].append(schedule)

    # Combine all years
    combined = {}
    for key, dfs in all_data.items():
        if dfs:
            combined[key] = pd.concat(dfs, ignore_index=True)
        else:
            combined[key] = pd.DataFrame()

    # Add reliability scores
    combined['reliability'] = api.calculate_reliability_scores(years)

    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(combined, f)
        logger.info(f"Data cached to {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    logger.info("Ergast data fetch complete!")
    return combined


