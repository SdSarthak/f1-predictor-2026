#!/usr/bin/env python
"""
F1 Race Predictor 2026 - Main Runner
Production-ready pipeline for F1 race predictions.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predictor import F1Predictor
from src.data.pipeline import F1DataPipeline
from src.regulations.rules_2026 import Regulations2026


def train_model(years: List[int] = None, model_type: str = 'xgboost') -> Dict:
    """
    Train the F1 prediction model with historical data.
    
    Args:
        years: List of years to include in training (default: 2022-2025)
        model_type: 'xgboost', 'random_forest', or 'ensemble'
    
    Returns:
        Training metrics dictionary
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]
    
    print("=" * 70)
    print("F1 Race Predictor 2026 - Model Training")
    print("=" * 70)
    print(f"\nTraining years: {years}")
    print(f"Model type: {model_type}")
    
    predictor = F1Predictor()
    
    # Build training data
    print("\n[1/2] Building training dataset...")
    df = predictor.build_training_data(years=years, save=True)
    print(f"      Dataset: {len(df)} rows, {len(df.columns)} features")
    
    # Train model
    print("\n[2/2] Training model...")
    metrics = predictor.train(df, model_type=model_type, save=True)
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"\nModel Performance:")
    print(f"  Mean Absolute Error: {metrics['mae']:.3f} positions")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  R² Score: {metrics['r2']:.3f}")
    print(f"\nPosition Accuracy:")
    print(f"  Within ±1 position: {metrics['within_1_position']*100:.1f}%")
    print(f"  Within ±2 positions: {metrics['within_2_positions']*100:.1f}%")
    print(f"  Within ±3 positions: {metrics['within_3_positions']*100:.1f}%")
    
    if 'cv_mae_mean' in metrics:
        print(f"\nCross-Validation:")
        print(f"  CV MAE: {metrics['cv_mae_mean']:.3f} (±{metrics['cv_mae_std']:.3f})")
    
    print(f"\nModel saved to: models/f1_predictor.joblib")
    
    return metrics


def predict_race(year: int = 2026,
                round_num: int = 1,
                circuit: str = None,
                grid: Dict[str, int] = None,
                simulations: int = 10000) -> Dict:
    """
    Predict a race result with Monte Carlo simulation.
    
    Args:
        year: Season year
        round_num: Round number
        circuit: Circuit name
        grid: Dict of driver_id -> grid position
        simulations: Number of Monte Carlo simulations
    
    Returns:
        Prediction results dictionary
    """
    print("=" * 70)
    print("F1 Race Predictor 2026 - Race Prediction")
    print("=" * 70)
    
    predictor = F1Predictor()
    
    # Load trained model
    model_path = Path("models/f1_predictor.joblib")
    if not model_path.exists():
        print("\nNo trained model found. Training with default settings...")
        train_model()
    
    predictor.load_model()
    
    # Make prediction
    print(f"\nPredicting: {year} Round {round_num}" + (f" - {circuit}" if circuit else ""))
    print(f"Monte Carlo simulations: {simulations:,}")
    
    results = predictor.predict_race(
        year=year,
        round_num=round_num,
        circuit_name=circuit,
        grid_positions=grid,
        run_simulation=True,
        n_simulations=simulations
    )
    
    # Display results
    print("\n" + predictor.format_predictions(results))
    
    # Save predictions
    predictor.save_predictions(results)
    
    return results


def show_2026_regulations():
    """Display 2026 regulation adjustments and team analysis."""
    print("\n" + "=" * 70)
    print("2026 F1 Regulation Analysis")
    print("=" * 70)
    
    regs = Regulations2026()
    
    teams = ['red_bull', 'mercedes', 'ferrari', 'mclaren', 'aston_martin', 
             'alpine', 'williams', 'rb', 'sauber', 'haas']
    
    # Team uncertainties
    print("\n1. Team Uncertainty Multipliers (Race 1 of 2026):")
    print("-" * 60)
    print(f"{'Team':20} {'Total':>10} {'Engine':>10} {'Driver':>10} {'Season':>10}")
    print("-" * 60)
    for team in teams:
        u = regs.calculate_2026_team_uncertainty(team, race_number=1)
        print(f"{team:20} {u['total']:>10.2f} {u['engine']:>10.2f} "
              f"{u['driver']:>10.2f} {u['season']:>10.2f}")
    
    # Power Unit profiles
    print("\n2. Power Unit Performance (Power-Hungry Tracks like Monza):")
    print("-" * 60)
    print(f"{'Team':20} {'Battery':>10} {'ICE':>10} {'Combined':>12}")
    print("-" * 60)
    for team in teams:
        pu = regs.calculate_power_unit_2026_profile(team, 'power_hungry')
        print(f"{team:20} {pu['battery_score']:>10.2f} {pu['ice_score']:>10.2f} "
              f"{pu['combined_pu_score']:>12.2f}")
    
    # Active Aero ratings
    print("\n3. Active Aero Efficiency Ratings (Based on Historical DRS):")
    print("-" * 40)
    for team_name, rating in sorted(regs.active_aero_baseline.items(), key=lambda x: -x[1]):
        bar = "█" * int(rating * 20)
        print(f"  {team_name:20} {rating:.2f} {bar}")
    
    # 2026 driver lineup
    print("\n4. Projected 2026 Driver Lineup:")
    print("-" * 60)
    lineup = regs.get_2026_driver_lineup()
    for team, drivers in lineup.items():
        print(f"  {team:20} | {drivers[0]:20} | {drivers[1]}")


def fetch_data(years: List[int] = None, save: bool = True):
    """
    Fetch and save F1 data for specified years.
    
    Args:
        years: List of years to fetch
        save: Whether to save the dataset
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]
    
    print("=" * 70)
    print("F1 Data Pipeline - Fetching Data")
    print("=" * 70)
    print(f"\nYears: {years}")
    
    pipeline = F1DataPipeline()
    
    print("\nFetching data from Ergast API and FastF1...")
    print("(This may take several minutes for multiple seasons)")
    
    df = pipeline.build_training_dataset(years)
    
    if save:
        pipeline.save_dataset(df)
        print(f"\nData saved to: data/training_data.parquet")
    
    print(f"\nDataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Years: {df['year'].unique().tolist()}")
    print(f"  Drivers: {df['driver_id'].nunique()}")
    print(f"  Races: {df.groupby(['year', 'round']).ngroups}")
    
    return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="F1 Race Predictor 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_predictor.py --train
  python run_predictor.py --train --years 2023 2024 2025
  python run_predictor.py --predict --round 1 --circuit Bahrain
  python run_predictor.py --regulations
  python run_predictor.py --fetch --years 2024 2025
        """
    )
    
    # Actions
    parser.add_argument('--train', action='store_true', 
                       help='Train the prediction model')
    parser.add_argument('--predict', action='store_true', 
                       help='Predict a race result')
    parser.add_argument('--regulations', action='store_true', 
                       help='Show 2026 regulation analysis')
    parser.add_argument('--fetch', action='store_true', 
                       help='Fetch and save F1 data')
    
    # Training options
    parser.add_argument('--years', type=int, nargs='+', default=None,
                       help='Years to include (default: 2022-2025)')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest', 'ensemble'],
                       help='Model type (default: xgboost)')
    
    # Prediction options
    parser.add_argument('--year', type=int, default=2026,
                       help='Prediction year (default: 2026)')
    parser.add_argument('--round', type=int, default=1,
                       help='Round number (default: 1)')
    parser.add_argument('--circuit', type=str, default=None,
                       help='Circuit name')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Monte Carlo simulations (default: 10000)')
    
    args = parser.parse_args()
    
    # Execute requested action
    if args.train:
        train_model(years=args.years, model_type=args.model)
    elif args.predict:
        predict_race(
            year=args.year,
            round_num=args.round,
            circuit=args.circuit,
            simulations=args.simulations
        )
    elif args.regulations:
        show_2026_regulations()
    elif args.fetch:
        fetch_data(years=args.years)
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("Quick Start:")
        print("=" * 70)
        print("\n1. Fetch data and train model:")
        print("   python run_predictor.py --train")
        print("\n2. Predict a race:")
        print("   python run_predictor.py --predict --circuit Bahrain")
        print("\n3. View 2026 regulations:")
        print("   python run_predictor.py --regulations")


if __name__ == "__main__":
    main()
