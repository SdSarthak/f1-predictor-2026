"""
F1 Race Predictor 2026 - Main Prediction Interface
Complete pipeline from data to predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import re
import yaml
import json

from src.constants import race_laps_for
from src.data.pipeline import F1DataPipeline
from src.features.feature_engineering import FeatureEngineer, engineer_features
from src.models.race_updater import RaceByRaceUpdater
from src.models.trainer import F1ModelTrainer
from src.simulation.monte_carlo import MonteCarloSimulator
from src.regulations.rules_2026 import Regulations2026

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1Predictor:
    """
    Main F1 Race Predictor for 2026.
    
    Combines ML predictions with Monte Carlo simulation
    and 2026 regulation adjustments.
    """
    
    def __init__(self,
                 config_path: str = "config/settings.yaml",
                 use_elo: bool = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Initialize components
        self.pipeline = F1DataPipeline(config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = F1ModelTrainer(config_path)
        self.simulator = MonteCarloSimulator(config_path)
        self.regulations = Regulations2026(config_path)

        elo_config = self.config.get('elo', {})
        self.use_elo = elo_config.get('enabled', True) if use_elo is None else use_elo
        self.race_updater = RaceByRaceUpdater(
            elo_weight=elo_config.get('weight', 0.40),
            k_factor_driver=elo_config.get('k_factor_driver', 20),
            k_factor_constructor=elo_config.get('k_factor_constructor', 40),
            ratings_path=elo_config.get('ratings_path', 'models/elo_ratings.json'),
        )

        self.is_trained = False
        self.training_data: Optional[pd.DataFrame] = None
        self.history_path = "data/training_data.parquet"

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML, falling back to built-in defaults."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}; using defaults")
            return {}
        except yaml.YAMLError as exc:
            logger.error(f"Could not parse {config_path}: {exc}; using defaults")
            return {}


    def build_training_data(self,
                            years: List[int] = None,
                            save: bool = True,
                            use_fastf1: Optional[bool] = None) -> pd.DataFrame:
        """
        Build complete training dataset.

        Args:
            years: Years to include in training data
            save: Whether to save the dataset
            use_fastf1: Include FastF1 telemetry features (slow); defaults to
                the `data.use_fastf1` config value

        Returns:
            Training DataFrame
        """
        if years is None:
            years = self.config.get('data', {}).get('years_to_fetch', [2022, 2023, 2024, 2025])

        logger.info(f"Building training data for years: {years}")

        # Fetch and merge data
        df = self.pipeline.build_training_dataset(years, use_fastf1=use_fastf1)


        # Engineer features
        df, self.feature_engineer = engineer_features(df, self.config)
        
        # Apply 2026 adjustments
        from src.regulations.rules_2026 import apply_2026_adjustments
        df = apply_2026_adjustments(df, self.config_path)
        
        self.training_data = df
        
        if save:
            self.pipeline.save_dataset(df)
        
        logger.info(f"Training data built: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load_history(self, path: Optional[str] = None, required: bool = False) -> bool:
        """
        Load the saved training dataset and refit the feature engineer on it.

        Prediction depends on this for two things that are otherwise silently
        missing: each driver's most recent engineered feature row, and the
        fitted label encoders for the categorical columns. Without it every
        driver is scored on neutral defaults with `*_encoded == -1`, i.e. the
        model degenerates into "predict the grid order".

        Returns True if history is available afterwards.
        """
        if self.training_data is not None and not self.training_data.empty:
            return True

        path = path or self.history_path
        try:
            df = self.pipeline.load_dataset(path)
        except (OSError, ValueError) as exc:
            message = (f"Could not load training history from {path}: {exc}. "
                       "Predictions will fall back to neutral feature values - "
                       "run `--train` (or `--fetch`) to rebuild it.")
            if required:
                raise FileNotFoundError(message) from exc
            logger.warning(message)
            return False

        if df is None or df.empty:
            logger.warning(f"Training history at {path} is empty; ignoring it")
            return False

        df, self.feature_engineer = engineer_features(df, self.config)
        self.training_data = df
        logger.info(f"Loaded {len(df)} historical rows from {path}")
        return True

    def train(self,
             df: pd.DataFrame = None,
             model_type: str = 'xgboost',
             save: bool = True) -> Dict[str, float]:
        """
        Train the prediction model.
        
        Args:
            df: Training DataFrame (if None, loads from file)
            model_type: 'xgboost', 'random_forest', or 'ensemble'
            save: Whether to save the trained model
            
        Returns:
            Training metrics
        """
        if df is None:
            self.load_history(required=True)
            df = self.training_data
        else:
            # Keep the engineer that produced `df` in sync with what is scored
            # later; `build_training_data` already set it.
            self.training_data = df


        # Get feature columns
        feature_columns = self.feature_engineer.get_feature_columns()
        target_column = self.feature_engineer.get_target_column()
        
        # Prepare data
        X, y = self.model_trainer.prepare_data(df, feature_columns, target_column)
        
        # Train
        metrics = self.model_trainer.train(X, y, model_type=model_type)
        
        if save:
            self.model_trainer.save_model()
        
        self.is_trained = True
        
        return metrics
    
    def load_model(self,
                   path: str = "models/f1_predictor.joblib",
                   with_history: bool = True):
        """
        Load a pre-trained model.

        Also reloads the training history by default. The model alone is not
        enough to score a race: the per-driver feature rows and the fitted
        categorical encoders live in the dataset, and without them every driver
        is scored on identical neutral defaults.
        """
        self.model_trainer.load_model(path)
        self.is_trained = True

        if with_history and not self.load_history():
            logger.warning(
                "Predicting without training history: driver form, constructor "
                "form, track record and the encoded categoricals will all sit at "
                "neutral defaults, so the result is little more than grid order."
            )

        logger.info("Model loaded successfully")
    
    def predict_race(self,
                    year: int = 2026,
                    round_num: int = 1,
                    circuit_name: str = None,
                    drivers: List[str] = None,
                    grid_positions: Dict[str, int] = None,
                    run_simulation: bool = True,
                    n_simulations: int = 10000,
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict a race result.
        
        Args:
            year: Season year
            round_num: Round number
            circuit_name: Name of the circuit
            drivers: List of driver IDs (if None, uses current grid)
            grid_positions: Qualifying positions (if None, uses prediction)
            run_simulation: Whether to run Monte Carlo simulation
            n_simulations: Number of MC simulations
            seed: Seed for the Monte Carlo stage, making the run reproducible
            
        Returns:
            Complete prediction results
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() or load_model() first.")
        
        logger.info(f"Predicting {year} Round {round_num}" + 
                   (f" ({circuit_name})" if circuit_name else ""))
        
        # Build features for prediction
        prediction_features = self._build_prediction_features(
            year, round_num, circuit_name, drivers, grid_positions
        )
        
        # Get ML predictions
        X = self._prepare_prediction_input(prediction_features)
        predictions, uncertainties = self.model_trainer.predict_with_uncertainty(X)
        
        # Map back to drivers
        driver_predictions = {}
        driver_uncertainties = {}
        constructor_map = {}
        driver_names = {}

        for i, (_, row) in enumerate(prediction_features.iterrows()):
            driver = row['driver_id']
            driver_predictions[driver] = float(predictions[i])
            driver_uncertainties[driver] = float(uncertainties[i])
            constructor_map[driver] = row['constructor_id']
            driver_names[driver] = row.get('driver_name', driver)

        # Blend in Elo form before the regulation layer, so the 2026 adjustments
        # apply to the combined ML + form view rather than the raw model output.
        elo_predictions = None
        if self.use_elo:
            blended, blended_unc = self.race_updater.blend_predictions(
                driver_predictions,
                driver_uncertainties,
                constructor_map,
                circuit=circuit_name,
            )
            elo_predictions = dict(driver_predictions)
            driver_predictions, driver_uncertainties = blended, blended_unc

        # Apply 2026 adjustments
        circuit_type = self._get_circuit_type(circuit_name)
        adjusted = self.regulations.adjust_predictions_for_2026(
            driver_predictions,
            driver_uncertainties,
            constructor_map,
            race_number=round_num,
            circuit_type=circuit_type
        )

        # The feature builder already reconciled any partial grid into a
        # complete one, so read it back rather than re-deriving it.
        grid = {
            row['driver_id']: int(row['grid_position'])
            for _, row in prediction_features.iterrows()
        }

        result = {
            'year': year,
            'round': round_num,
            'circuit': circuit_name,
            'circuit_type': circuit_type,
            'drivers': driver_names,
            'constructors': constructor_map,
            'grid_positions': grid,
            'raw_predictions': elo_predictions if elo_predictions is not None else driver_predictions,
            'blended_predictions': driver_predictions,
            'adjusted_predictions': adjusted['predictions'],
            'uncertainties': adjusted['uncertainties'],
            '2026_adjustments': adjusted['adjustments'],
            'elo_weight': (self.race_updater.elo_weight
                           if self.use_elo and self.race_updater.has_ratings else 0.0),
        }

        # Run Monte Carlo simulation
        if run_simulation:
            reliability = self._get_reliability_scores(
                list(adjusted['predictions'].keys()), constructor_map
            )

            mc_results = self.simulator.run_monte_carlo(
                adjusted['predictions'],
                adjusted['uncertainties'],
                grid,
                reliability,
                race_laps=self._get_race_laps(circuit_name),
                circuit_type=circuit_type,
                n_simulations=n_simulations,
                seed=seed,
            )
            
            result['monte_carlo'] = {
                'win_probabilities': mc_results['win_probabilities'],
                'podium_probabilities': mc_results['podium_probabilities'],
                'points_probabilities': mc_results['points_probabilities'],
                'expected_positions': mc_results['expected_positions'],
                'position_std': mc_results['position_std'],
                'dnf_probabilities': mc_results['dnf_probability'],
            }
            
            # Get confidence intervals
            intervals = self.simulator.get_confidence_intervals(
                mc_results['position_distributions']
            )
            result['confidence_intervals'] = intervals
        
        return result
    
    # Features that describe the car, not the driver. When a driver changes team
    # these must come from the new constructor's history, otherwise a driver who
    # moved (Tsunoda: Red Bull -> RB) carries his old team's car performance.
    CONSTRUCTOR_SCOPED_FEATURES = (
        'constructor_form',
        'reliability_score',
        'pit_efficiency',
        'pit_consistency',
        'active_aero_efficiency',
        'season_avg_deg',
        'is_new_engine',
    )

    def _default_feature_value(self, column: str) -> float:
        """Neutral fallback for a feature column (shared with the engineer)."""
        return FeatureEngineer.default_for(column)

    def _resolve_grid_entries(self,
                              year: int,
                              drivers: Optional[List[str]]) -> List[Dict[str, str]]:
        """
        Work out which driver/constructor pairs to predict for.

        Preference order:
          1. the explicit ``drivers`` argument,
          2. the confirmed 2026 entry list for 2026+ races,
          3. the most recent season present in the loaded training data.

        The 2026 list wins over history for 2026 races because a season's
        history contains every driver who *appeared*, including mid-season
        replacements - 2025 alone yields 22 entries for 20 seats.
        """
        entries: List[Dict[str, str]] = []

        if year >= 2026:
            entries = self.regulations.get_2026_grid_entries()

        if not entries and self.training_data is not None and not self.training_data.empty:
            history = self.training_data
            latest_season = history[history['year'] == history['year'].max()]

            # Only the cars that took the start of the final round: a season's
            # full driver list includes anyone who was replaced mid-year, which
            # would put more cars on the grid than there are seats.
            final_round = latest_season[latest_season['round'] == latest_season['round'].max()]
            if not final_round.empty:
                latest_season = latest_season[
                    latest_season['driver_id'].isin(final_round['driver_id'])
                ]

            latest = latest_season.groupby('driver_id').last().reset_index()

            for _, row in latest.iterrows():
                entries.append({
                    'driver_id': row['driver_id'],
                    'driver_code': row.get('driver_code') or str(row['driver_id'])[:3].upper(),
                    'driver_name': row.get('driver_name') or row['driver_id'],
                    'constructor_id': row.get('constructor_id', 'unknown'),
                    'constructor_name': row.get('constructor_name', 'Unknown'),
                })

        if not entries:
            entries = self.regulations.get_2026_grid_entries()

        if drivers:
            wanted = {d.lower() for d in drivers}
            filtered = [e for e in entries if e['driver_id'].lower() in wanted]

            # Any requested driver missing from history still gets an entry so
            # the caller always receives a prediction for what they asked for.
            known = {e['driver_id'].lower() for e in filtered}
            lineup_2026 = {e['driver_id'].lower(): e
                           for e in self.regulations.get_2026_grid_entries()}
            for driver in drivers:
                key = driver.lower()
                if key in known:
                    continue
                filtered.append(lineup_2026.get(key, {
                    'driver_id': driver,
                    'driver_code': driver[:3].upper(),
                    'driver_name': driver,
                    'constructor_id': 'unknown',
                    'constructor_name': 'Unknown',
                }))
            entries = filtered

        return entries

    def _estimate_grid_positions(self, entries: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Estimate a starting grid when qualifying results are not supplied.

        Drivers are ranked by their average historical grid position where that
        is known, and by 2026 pre-season/opening-round pace where it is not.
        """
        history = self.training_data
        scores: Dict[str, float] = {}

        for entry in entries:
            driver_id = entry['driver_id']
            score = None

            if history is not None and not history.empty and 'grid_position' in history.columns:
                driver_rows = history[history['driver_id'] == driver_id]
                if not driver_rows.empty:
                    mean_grid = driver_rows['grid_position'].tail(10).mean()
                    if pd.notna(mean_grid):
                        score = float(mean_grid)

            if score is None:
                # Lower rating = further back, so invert into a pseudo grid slot.
                team_name = self.regulations.get_constructor_display_name(
                    entry.get('constructor_id', '')
                )
                rating = self.regulations.testing_data.get_testing_performance_rating(team_name)
                score = 21.0 - rating * 20.0

            scores[driver_id] = score

        ordered = sorted(scores.items(), key=lambda item: item[1])
        return {driver_id: position for position, (driver_id, _) in enumerate(ordered, 1)}

    def _resolve_grid(self,
                      entries: List[Dict[str, str]],
                      grid_positions: Optional[Dict[str, int]]) -> Dict[str, int]:
        """
        Produce a complete grid.

        Known qualifying results are kept exactly as given; anyone not listed is
        slotted into the remaining positions in estimated-pace order, so a
        partial grid (say, just the front row) still yields a full field.
        """
        estimated = self._estimate_grid_positions(entries)

        if not grid_positions:
            return estimated

        grid: Dict[str, int] = {}
        taken = set()
        for driver_id, position in grid_positions.items():
            position = int(position)
            grid[driver_id] = position
            taken.add(position)

        unplaced = [entry['driver_id'] for entry in entries if entry['driver_id'] not in grid]
        unplaced.sort(key=lambda driver: estimated.get(driver, len(entries)))

        free_slots = (p for p in range(1, len(entries) + 1) if p not in taken)
        for driver_id in unplaced:
            grid[driver_id] = next(free_slots, len(entries))

        return grid

    def _circuit_flags(self, circuit_name: Optional[str]) -> Dict[str, int]:
        """Binary track-category flags for the circuit being predicted."""
        track_cats = self.config.get('track_categories', {})
        name = (circuit_name or '').lower()

        flags = {
            f'is_{category}': int(any(track.lower() in name for track in tracks))
            for category, tracks in track_cats.items()
        }
        flags['is_power_track'] = flags.get('is_power_hungry', 0)
        return flags

    def _lookup_circuit_id(self, circuit_name: Optional[str]) -> str:
        """Best-effort mapping from a circuit name to its historical id."""
        if not circuit_name:
            return 'unknown'

        history = self.training_data
        if history is not None and 'circuit_id' in history.columns:
            name = circuit_name.lower()
            matches = history[
                history['circuit_name'].astype(str).str.lower().str.contains(name, regex=False)
                | history['circuit_id'].astype(str).str.lower().str.contains(name, regex=False)
            ]
            if not matches.empty:
                return str(matches.iloc[-1]['circuit_id'])

        return circuit_name.lower().replace(' ', '_')

    def _build_prediction_features(self,
                                   year: int,
                                   round_num: int,
                                   circuit_name: str,
                                   drivers: List[str],
                                   grid_positions: Dict[str, int]) -> pd.DataFrame:
        """
        Build the feature DataFrame the model scores for an upcoming race.

        Each driver's most recent engineered feature row is carried forward, then
        the race-specific values (grid slot, circuit flags) are overwritten.
        """
        entries = self._resolve_grid_entries(year, drivers)
        if not entries:
            raise ValueError("No drivers available to predict. Provide `drivers` explicitly.")

        feature_columns = self.feature_engineer.get_feature_columns()
        history = self.training_data
        circuit_id = self._lookup_circuit_id(circuit_name)
        circuit_flags = self._circuit_flags(circuit_name)
        grid = self._resolve_grid(entries, grid_positions)
        rain_ratings = self.config.get('rain_performance', {})
        default_rain = rain_ratings.get('default', 0.75)

        rows = []
        for entry in entries:
            driver_id = entry['driver_id']

            row: Dict[str, Any] = {
                'driver_id': driver_id,
                'driver_code': entry['driver_code'],
                'driver_name': entry['driver_name'],
                'constructor_id': entry['constructor_id'],
                'constructor_name': entry['constructor_name'],
                'circuit_id': circuit_id,
                'circuit_name': circuit_name or 'Unknown',
                'year': year,
                'round': round_num,
            }

            # Carry forward the driver's latest engineered features.
            if history is not None and not history.empty:
                driver_history = history[history['driver_id'] == driver_id]

                # Ergast ids and the 2026 entry list disagree for a handful of
                # drivers (`max_verstappen` vs `verstappen`); the three-letter
                # code is stable across both, so fall back to it.
                if driver_history.empty and 'driver_code' in history.columns:
                    code = entry.get('driver_code')
                    if code:
                        driver_history = history[history['driver_code'] == code]

                if not driver_history.empty:
                    latest = driver_history.iloc[-1]
                    # Encode against the id the label encoder was fitted on.
                    row['_history_driver_id'] = latest['driver_id']
                    for column in feature_columns:
                        if column in latest.index and pd.notna(latest[column]):
                            row[column] = latest[column]

                    # Track experience is cumulative, so advance it by one race.
                    track_rows = driver_history[driver_history['circuit_id'] == circuit_id]
                    row['track_experience'] = float(len(track_rows))
                    if not track_rows.empty and 'track_historical_perf' in track_rows.columns:
                        value = track_rows.iloc[-1]['track_historical_perf']
                        if pd.notna(value):
                            row['track_historical_perf'] = value

            # Car-level features come from the 2026 constructor, not from
            # whichever team the driver last raced for.
            if history is not None and not history.empty:
                team_history = history[history['constructor_id'] == entry['constructor_id']]
                if not team_history.empty:
                    team_latest = team_history.iloc[-1]
                    for column in self.CONSTRUCTOR_SCOPED_FEATURES:
                        if column in team_latest.index and pd.notna(team_latest[column]):
                            row[column] = team_latest[column]

            # Race-specific values always win over carried-forward history.
            row.update(circuit_flags)
            row['grid_position'] = float(grid.get(driver_id, len(entries)))
            row['is_front_row'] = int(row['grid_position'] <= 2)
            row['is_top5_grid'] = int(row['grid_position'] <= 5)
            row['is_top10_grid'] = int(row['grid_position'] <= 10)
            row['rain_skill'] = rain_ratings.get(entry['driver_code'], default_rain)
            row['historical_weight'] = self.config.get(
                'regulations_2026', {}
            ).get('historical_weight_decay', {}).get(year, 0.4)

            rows.append(row)

        df = pd.DataFrame(rows)

        # Encode with the historical driver id where the two vocabularies differ,
        # then restore the entry-list id for reporting.
        display_ids = df['driver_id'].copy()
        if '_history_driver_id' in df.columns:
            df['driver_id'] = df['_history_driver_id'].fillna(df['driver_id'])
        df = self.feature_engineer.transform_categoricals(df)
        df['driver_id'] = display_ids
        df = df.drop(columns=['_history_driver_id'], errors='ignore')

        # Guarantee every model input exists and is finite.
        for column in feature_columns:
            default = self._default_feature_value(column)
            if column not in df.columns:
                df[column] = default
            else:
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(default)

        return df

    def _prepare_prediction_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare the feature matrix, column-aligned with the trained model."""
        expected = self.model_trainer.feature_columns
        if not expected:
            raise ValueError("Trained model has no feature columns recorded.")

        X = pd.DataFrame(index=df.index)
        for column in expected:
            if column in df.columns:
                X[column] = pd.to_numeric(df[column], errors='coerce')
            else:
                logger.warning(f"Feature '{column}' missing at prediction time; using default")
                X[column] = self._default_feature_value(column)

        X = X.replace([np.inf, -np.inf], np.nan)
        for column in expected:
            X[column] = X[column].fillna(self._default_feature_value(column))

        return X.values.astype(float)


    def _get_circuit_type(self, circuit_name: str) -> str:
        """Determine circuit type from name."""
        if circuit_name is None:
            return 'normal'
        
        track_cats = self.config.get('track_categories', {})
        
        for category, tracks in track_cats.items():
            if any(t.lower() in circuit_name.lower() for t in tracks):
                return category
        
        return 'normal'
    
    def _get_reliability_scores(self,
                                drivers: List[str],
                                constructor_map: Dict[str, str] = None) -> Dict[str, float]:
        """
        Reliability (1 - DNF rate) per driver, taken from their team.

        Prefers the 2026 team reliability table in the config, then the DNF rate
        observed in the training data, then a neutral default.
        """
        constructor_map = constructor_map or {}

        config_reliability = self.config.get('team_reliability_2026', {})

        historical: Dict[str, float] = {}
        history = self.training_data
        if history is not None and 'reliability_score' in history.columns:
            historical = history.groupby('constructor_id')['reliability_score'].mean().to_dict()

        driver_reliability = {}
        for driver in drivers:
            constructor_id = constructor_map.get(driver, '')
            display_name = self.regulations.get_constructor_display_name(constructor_id)

            score = config_reliability.get(display_name)
            if score is None:
                score = historical.get(constructor_id)
            if score is None or not np.isfinite(score):
                score = 0.92

            driver_reliability[driver] = float(np.clip(score, 0.5, 1.0))

        return driver_reliability


    def _get_race_laps(self, circuit_name: str) -> int:
        """Get number of race laps for a circuit."""
        return race_laps_for(circuit_name)

    def update_after_race(self,
                          results: pd.DataFrame,
                          circuit: str,
                          round_num: int) -> pd.DataFrame:
        """
        Fold a completed race into the Elo ratings.

        Args:
            results: DataFrame with `driver_id`, `constructor_id`, `position`
                and optionally `quali_position` and `dnf`.
            circuit: Circuit name.
            round_num: Round number within the season.

        Returns:
            The updated rating table.
        """
        required = {'driver_id', 'constructor_id', 'position'}
        missing = required - set(results.columns)
        if missing:
            raise ValueError(f"Race results are missing required columns: {sorted(missing)}")

        self.race_updater.update_from_race(results, circuit=circuit, race_number=round_num)
        return self.race_updater.get_rating_summary()

    def get_elo_standings(self) -> pd.DataFrame:
        """Current Elo ratings for drivers and constructors."""
        return self.race_updater.get_rating_summary()

    def format_predictions(self, results: Dict[str, Any]) -> str:
        """Format prediction results for display."""
        names = results.get('drivers', {})
        constructors = results.get('constructors', {})
        grid = results.get('grid_positions', {})

        def label(driver_id: str) -> str:
            name = names.get(driver_id, driver_id)
            team = constructors.get(driver_id)
            if team:
                return f"{name} ({self.regulations.get_constructor_display_name(team)})"
            return str(name)

        output = []
        output.append("=" * 78)
        output.append(f"F1 {results['year']} Race Prediction: {results.get('circuit') or 'Unknown'}")
        output.append(f"Round {results['round']} | Circuit Type: {results['circuit_type']}"
                      + (f" | Elo weight: {results['elo_weight']:.0%}"
                         if results.get('elo_weight') else ""))
        output.append("=" * 78)

        if 'monte_carlo' in results:
            mc = results['monte_carlo']

            # Sort by expected position
            sorted_drivers = sorted(
                mc['expected_positions'].items(),
                key=lambda x: x[1]
            )

            output.append("\n{:4} {:32} {:>5} {:>9} {:>10} {:>9} {:>7}".format(
                "Pos", "Driver", "Grid", "Win %", "Podium %", "Exp Pos", "+/-Std"
            ))
            output.append("-" * 78)

            for i, (driver, exp_pos) in enumerate(sorted_drivers, 1):
                win_pct = mc['win_probabilities'].get(driver, 0) * 100
                podium_pct = mc['podium_probabilities'].get(driver, 0) * 100
                std = mc['position_std'].get(driver, 0)
                grid_slot = grid.get(driver, '-')

                output.append("{:<4} {:32} {:>5} {:>8.1f}% {:>9.1f}% {:>9.1f} {:>7.1f}".format(
                    i, label(driver)[:32], grid_slot, win_pct, podium_pct, exp_pos, std
                ))

            # Confidence intervals
            if 'confidence_intervals' in results:
                output.append("\n90% Confidence Intervals (top 10):")
                output.append("-" * 50)
                for driver, (low, high) in sorted(
                    results['confidence_intervals'].items(),
                    key=lambda x: results['monte_carlo']['expected_positions'][x[0]]
                )[:10]:
                    output.append(f"  {label(driver):40} P{low} - P{high}")
        else:
            # Simple predictions
            sorted_preds = sorted(
                results['adjusted_predictions'].items(),
                key=lambda x: x[1]
            )

            output.append("\nPredicted Order:")
            for i, (driver, pos) in enumerate(sorted_preds, 1):
                output.append(f"  {i:2}. {label(driver):40} (predicted: {pos:.1f})")

        return "\n".join(output)


    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """Recursively convert numpy scalars/arrays into JSON-safe values."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): F1Predictor._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [F1Predictor._to_jsonable(v) for v in obj]
        return obj

    def save_predictions(self, results: Dict[str, Any], path: str = "predictions/") -> Path:
        """
        Write prediction results to a JSON file.

        The full position distributions are dropped - they are megabytes of
        simulation detail that the summary statistics already capture.
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        circuit = results.get('circuit') or 'race'
        safe_circuit = re.sub(r'[^A-Za-z0-9._-]+', '_', str(circuit)).strip('_') or 'race'
        filepath = output_dir / f"{results['year']}_R{results['round']}_{safe_circuit}.json"

        payload = {k: v for k, v in results.items() if k != 'position_distributions'}
        clean_results = self._to_jsonable(payload)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2)

        logger.info(f"Predictions saved to {filepath}")
        return filepath
