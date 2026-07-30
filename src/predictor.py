"""
F1 Race Predictor 2026 - Main Prediction Interface
Complete pipeline from data to predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import yaml
import json
from datetime import datetime

from src.data.pipeline import F1DataPipeline
from src.data.ergast_api import ErgastAPI
from src.data.fastf1_client import FastF1Client
from src.features.feature_engineering import FeatureEngineer, engineer_features
from src.models.trainer import F1ModelTrainer, train_model
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
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.pipeline = F1DataPipeline(config_path)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = F1ModelTrainer(config_path)
        self.simulator = MonteCarloSimulator(config_path)
        self.regulations = Regulations2026(config_path)
        
        self.is_trained = False
        self.training_data: Optional[pd.DataFrame] = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {}
    
    def build_training_data(self, years: List[int] = None, save: bool = True) -> pd.DataFrame:
        """
        Build complete training dataset.
        
        Args:
            years: Years to include in training data
            save: Whether to save the dataset
            
        Returns:
            Training DataFrame
        """
        if years is None:
            years = self.config.get('data', {}).get('years_to_fetch', [2022, 2023, 2024, 2025])
        
        logger.info(f"Building training data for years: {years}")
        
        # Fetch and merge data
        df = self.pipeline.build_training_dataset(years)
        
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
            if self.training_data is not None:
                df = self.training_data
            else:
                df = self.pipeline.load_dataset()
                df, self.feature_engineer = engineer_features(df, self.config)
        
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
    
    def load_model(self, path: str = "models/f1_predictor.joblib"):
        """Load a pre-trained model."""
        self.model_trainer.load_model(path)
        self.is_trained = True
        logger.info("Model loaded successfully")
    
    def predict_race(self,
                    year: int = 2026,
                    round_num: int = 1,
                    circuit_name: str = None,
                    drivers: List[str] = None,
                    grid_positions: Dict[str, int] = None,
                    run_simulation: bool = True,
                    n_simulations: int = 10000) -> Dict[str, Any]:
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
        
        for i, row in prediction_features.iterrows():
            driver = row['driver_id']
            driver_predictions[driver] = predictions[i]
            driver_uncertainties[driver] = uncertainties[i]
            constructor_map[driver] = row['constructor_id']
        
        # Apply 2026 adjustments
        circuit_type = self._get_circuit_type(circuit_name)
        adjusted = self.regulations.adjust_predictions_for_2026(
            driver_predictions,
            driver_uncertainties,
            constructor_map,
            race_number=round_num,
            circuit_type=circuit_type
        )
        
        result = {
            'year': year,
            'round': round_num,
            'circuit': circuit_name,
            'circuit_type': circuit_type,
            'raw_predictions': driver_predictions,
            'adjusted_predictions': adjusted['predictions'],
            'uncertainties': adjusted['uncertainties'],
            '2026_adjustments': adjusted['adjustments'],
        }
        
        # Run Monte Carlo simulation
        if run_simulation:
            grid = grid_positions or {d: i+1 for i, d in enumerate(adjusted['predictions'].keys())}
            reliability = self._get_reliability_scores(list(adjusted['predictions'].keys()))
            
            mc_results = self.simulator.run_monte_carlo(
                adjusted['predictions'],
                adjusted['uncertainties'],
                grid,
                reliability,
                race_laps=self._get_race_laps(circuit_name),
                circuit_type=circuit_type,
                n_simulations=n_simulations
            )
            
            result['monte_carlo'] = {
                'win_probabilities': mc_results['win_probabilities'],
                'podium_probabilities': mc_results['podium_probabilities'],
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
    
    def _build_prediction_features(self,
                                   year: int,
                                   round_num: int,
                                   circuit_name: str,
                                   drivers: List[str],
                                   grid_positions: Dict[str, int]) -> pd.DataFrame:
        """Build feature DataFrame for prediction."""
        # Get latest driver/team data
        if self.training_data is not None:
            latest_data = self.training_data[
                self.training_data['year'] == self.training_data['year'].max()
            ]
        else:
            # Fetch latest data
            api = ErgastAPI()
            latest_data = api.get_race_results(year - 1)
        
        # Get unique driver-constructor pairs
        if drivers is None:
            driver_teams = latest_data.groupby('driver_id').last()[
                ['constructor_id', 'constructor_name', 'driver_code', 'driver_name']
            ].reset_index()
        else:
            driver_teams = latest_data[latest_data['driver_id'].isin(drivers)].groupby('driver_id').last()[
                ['constructor_id', 'constructor_name', 'driver_code', 'driver_name']
            ].reset_index()
        
        # Build prediction rows
        rows = []
        for _, dt in driver_teams.iterrows():
            row = {
                'driver_id': dt['driver_id'],
                'driver_code': dt.get('driver_code', dt['driver_id'][:3].upper()),
                'driver_name': dt.get('driver_name', dt['driver_id']),
                'constructor_id': dt['constructor_id'],
                'constructor_name': dt['constructor_name'],
                'year': year,
                'round': round_num,
                'circuit_name': circuit_name or 'Unknown',
            }
            
            # Add grid position if available
            if grid_positions and dt['driver_id'] in grid_positions:
                row['grid_position'] = grid_positions[dt['driver_id']]
            else:
                row['grid_position'] = len(driver_teams) // 2  # Mid-grid default
            
            # Get driver's historical features from training data
            if self.training_data is not None:
                driver_history = self.training_data[
                    self.training_data['driver_id'] == dt['driver_id']
                ]
                if not driver_history.empty:
                        latest = driver_history.iloc[-1]
        
        df = pd.DataFrame(rows)
        
        # Fill missing features
        for col in self.feature_engineer.get_feature_columns():
            if col not in df.columns:
                df[col] = 0.5 if 'rate' in col or 'score' in col else 0
        
        return df
    
    def _prepare_prediction_input(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for prediction."""
        feature_cols = [c for c in self.model_trainer.feature_columns if c in df.columns]
        X = df[feature_cols].copy()
        X = X.fillna(X.mean())
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        return X.values
    
    def _get_circuit_type(self, circuit_name: str) -> str:
        """Determine circuit type from name."""
        if circuit_name is None:
            return 'normal'
        
        track_cats = self.config.get('track_categories', {})
        
        for category, tracks in track_cats.items():
            if any(t.lower() in circuit_name.lower() for t in tracks):
                return category
        
        return 'normal'
    
    def _get_reliability_scores(self, drivers: List[str]) -> Dict[str, float]:
        """Get reliability scores for drivers."""
        if self.training_data is not None:
            reliability = self.training_data.groupby('constructor_id')['reliability_score'].mean().to_dict()
        else:
            reliability = {}
        
        # Map drivers to their team's reliability
        driver_reliability = {}
        for driver in drivers:
            # Default high reliability
            driver_reliability[driver] = reliability.get(driver, 0.92)
        
        return driver_reliability
    
    def _get_race_laps(self, circuit_name: str) -> int:
        """Get number of race laps for a circuit."""
        # Default race laps by circuit
        lap_counts = {
            'bahrain': 57,
            'jeddah': 50,
            'melbourne': 58,
            'suzuka': 53,
            'monaco': 78,
            'canada': 70,
            'silverstone': 52,
            'spa': 44,
            'monza': 53,
            'singapore': 62,
            'austin': 56,
            'las vegas': 50,
            'abu dhabi': 58,
        }
        
        if circuit_name:
            for key, laps in lap_counts.items():
                if key in circuit_name.lower():
                    return laps
        
        return 57  # Default
    
    def format_predictions(self, results: Dict[str, Any]) -> str:
        """Format prediction results for display."""
        output = []
        output.append("=" * 70)
        output.append(f"F1 2026 Race Prediction: {results.get('circuit', 'Unknown')}")
        output.append(f"Round {results['round']} | Circuit Type: {results['circuit_type']}")
        output.append("=" * 70)
        
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            
            # Sort by expected position
            sorted_drivers = sorted(
                mc['expected_positions'].items(),
                key=lambda x: x[1]
            )
            
            output.append("\n{:4} {:20} {:>10} {:>10} {:>10} {:>8}".format(
                "Pos", "Driver", "Win %", "Podium %", "Exp Pos", "±Std"
            ))
            output.append("-" * 70)
            
            for i, (driver, exp_pos) in enumerate(sorted_drivers, 1):
                win_pct = mc['win_probabilities'].get(driver, 0) * 100
                podium_pct = mc['podium_probabilities'].get(driver, 0) * 100
                std = mc['position_std'].get(driver, 0)
                
                output.append("{:4} {:20} {:>9.1f}% {:>9.1f}% {:>10.1f} {:>7.1f}".format(
                    i, driver, win_pct, podium_pct, exp_pos, std
                ))
            
            # Confidence intervals
            if 'confidence_intervals' in results:
                output.append("\n90% Confidence Intervals:")
                output.append("-" * 40)
                for driver, (low, high) in sorted(
                    results['confidence_intervals'].items(),
                    key=lambda x: results['monte_carlo']['expected_positions'][x[0]]
                )[:10]:
                    output.append(f"  {driver}: P{low} - P{high}")
        else:
            # Simple predictions
            sorted_preds = sorted(
                results['adjusted_predictions'].items(),
                key=lambda x: x[1]
            )
            
            output.append("\nPredicted Order:")
            for i, (driver, pos) in enumerate(sorted_preds, 1):
                output.append(f"  {i}. {driver} (predicted: {pos:.1f})")
        
        return "\n".join(output)
    
    def save_predictions(self, results: Dict[str, Any], path: str = "predictions/"):
        """Save predictions to file."""
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{results['year']}_R{results['round']}_{results.get('circuit', 'race')}.json"
        filepath = output_dir / filename
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        clean_results = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Predictions saved to {filepath}")


def main():
    """Main entry point for the predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="F1 2026 Race Predictor")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Make predictions')
    parser.add_argument('--year', type=int, default=2026, help='Prediction year')
    parser.add_argument('--round', type=int, default=1, help='Round number')
    parser.add_argument('--circuit', type=str, default=None, help='Circuit name')
    parser.add_argument('--simulations', type=int, default=10000, help='MC simulations')
    
    args = parser.parse_args()
    
    predictor = F1Predictor()
    
    if args.train:
        # Build training data and train
        print("Building training data...")
        df = predictor.build_training_data(years=[2022, 2023, 2024, 2025])
        
        print("Training model...")
        metrics = predictor.train(df, model_type='xgboost')
        
        print(f"\nTraining complete!")
        print(f"MAE: {metrics['mae']:.3f}")
        print(f"Within ±2 positions: {metrics['within_2_positions']*100:.1f}%")
    
    if args.predict:
        # Load model if not trained
        if not predictor.is_trained:
            predictor.load_model()
        
        # Make prediction
        results = predictor.predict_race(
            year=args.year,
            round_num=args.round,
            circuit_name=args.circuit,
            n_simulations=args.simulations
        )
        
        # Display results
        print(predictor.format_predictions(results))
        
        # Save
        predictor.save_predictions(results)


if __name__ == "__main__":
    main()
