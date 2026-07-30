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


def train_model(years: List[int] = None,
                model_type: str = 'xgboost',
                use_fastf1: Optional[bool] = None) -> Dict:
    """
    Train the F1 prediction model with historical data.

    Args:
        years: List of years to include in training (default: 2022-2025)
        model_type: 'xgboost', 'random_forest', or 'ensemble'
        use_fastf1: Include FastF1 telemetry features (slow)

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
    df = predictor.build_training_data(years=years, save=True, use_fastf1=use_fastf1)
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


def parse_grid(entries: Optional[List[str]]) -> Optional[Dict[str, int]]:
    """
    Parse `--grid driver=position` arguments into a grid mapping.

    Example: `--grid verstappen=1 norris=2`
    """
    if not entries:
        return None

    grid: Dict[str, int] = {}
    for entry in entries:
        if '=' not in entry:
            raise ValueError(f"Invalid --grid entry '{entry}'. Use driver_id=position.")
        driver, _, position = entry.partition('=')
        try:
            grid[driver.strip()] = int(position)
        except ValueError:
            raise ValueError(f"Grid position for '{driver}' must be an integer, got '{position}'.")

    return grid


def predict_race(year: int = 2026,
                round_num: int = 1,
                circuit: str = None,
                grid: Dict[str, int] = None,
                simulations: int = 10000,
                use_elo: bool = True) -> Dict:
    """
    Predict a race result with Monte Carlo simulation.

    Args:
        year: Season year
        round_num: Round number
        circuit: Circuit name
        grid: Dict of driver_id -> grid position
        simulations: Number of Monte Carlo simulations
        use_elo: Blend the ML output with race-by-race Elo form

    Returns:
        Prediction results dictionary
    """
    print("=" * 70)
    print("F1 Race Predictor 2026 - Race Prediction")
    print("=" * 70)

    predictor = F1Predictor(use_elo=use_elo)

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


def show_ratings() -> None:
    """Display the current Elo ratings for drivers and constructors."""
    predictor = F1Predictor()
    ratings = predictor.get_elo_standings()

    print("=" * 70)
    print("Elo Ratings")
    print("=" * 70)

    if ratings.empty:
        print("\nNo ratings yet. Run --update-race to seed them from a result.")
        return

    for kind, title in (('constructor', 'Constructors'), ('driver', 'Drivers')):
        subset = ratings[ratings['type'] == kind]
        if subset.empty:
            continue
        print(f"\n{title}:")
        print("-" * 40)
        for _, row in subset.iterrows():
            print(f"  {row['entity']:24} {row['rating']:8.1f}")


def update_from_results(path: str, circuit: str, round_num: int) -> None:
    """
    Fold a finished race into the Elo ratings.

    The CSV needs `driver_id`, `constructor_id` and `position` columns, plus
    optional `quali_position` and `dnf`.
    """
    import pandas as pd

    results = pd.read_csv(path)

    print("=" * 70)
    print(f"Updating Elo ratings from {path}")
    print("=" * 70)

    predictor = F1Predictor()
    ratings = predictor.update_after_race(results, circuit=circuit, round_num=round_num)

    constructors = ratings[ratings['type'] == 'constructor']
    print("\nUpdated constructor ratings:")
    print("-" * 40)
    for _, row in constructors.iterrows():
        print(f"  {row['entity']:24} {row['rating']:8.1f}")


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


def fetch_data(years: List[int] = None,
               save: bool = True,
               use_fastf1: Optional[bool] = None):
    """
    Fetch and save F1 data for specified years.

    Args:
        years: List of years to fetch
        save: Whether to save the dataset
        use_fastf1: Include FastF1 telemetry (slow)
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    print("=" * 70)
    print("F1 Data Pipeline - Fetching Data")
    print("=" * 70)
    print(f"\nYears: {years}")

    pipeline = F1DataPipeline()

    print("\nFetching data from the Ergast mirror" + (" and FastF1..." if use_fastf1 else "..."))
    print("(This may take several minutes for multiple seasons)")

    df = pipeline.build_training_dataset(years, use_fastf1=use_fastf1)


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
  python run_predictor.py --train --years 2023 2024 2025 --telemetry
  python run_predictor.py --predict --round 1 --circuit Bahrain
  python run_predictor.py --predict --circuit Monaco --grid verstappen=1 norris=2
  python run_predictor.py --regulations
  python run_predictor.py --ratings
  python run_predictor.py --update-race results.csv --circuit Bahrain --round 2
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
    parser.add_argument('--ratings', action='store_true',
                       help='Show current Elo ratings')
    parser.add_argument('--update-race', type=str, default=None, metavar='CSV',
                       help='Fold a finished race result (CSV) into the Elo ratings')

    # Training options
    parser.add_argument('--years', type=int, nargs='+', default=None,
                       help='Years to include (default: 2022-2025)')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'random_forest', 'ensemble'],
                       help='Model type (default: xgboost)')
    parser.add_argument('--telemetry', action='store_true',
                       help='Include FastF1 telemetry features (much slower)')

    # Prediction options
    parser.add_argument('--year', type=int, default=2026,
                       help='Prediction year (default: 2026)')
    parser.add_argument('--round', type=int, default=1,
                       help='Round number (default: 1)')
    parser.add_argument('--circuit', type=str, default=None,
                       help='Circuit name')
    parser.add_argument('--grid', type=str, nargs='+', default=None, metavar='DRIVER=POS',
                       help='Known qualifying results, e.g. --grid russell=1 norris=2')
    parser.add_argument('--simulations', type=int, default=10000,
                       help='Monte Carlo simulations (default: 10000)')
    parser.add_argument('--no-elo', action='store_true',
                       help='Use the raw ML prediction without the Elo form blend')

    args = parser.parse_args()
    use_fastf1 = True if args.telemetry else None

    # Execute requested action
    if args.train:
        train_model(years=args.years, model_type=args.model, use_fastf1=use_fastf1)
    elif args.predict:
        predict_race(
            year=args.year,
            round_num=args.round,
            circuit=args.circuit,
            grid=parse_grid(args.grid),
            simulations=args.simulations,
            use_elo=not args.no_elo,
        )
    elif args.regulations:
        show_2026_regulations()
    elif args.ratings:
        show_ratings()
    elif args.update_race:
        update_from_results(args.update_race, circuit=args.circuit or 'Unknown',
                            round_num=args.round)
    elif args.fetch:
        fetch_data(years=args.years, use_fastf1=use_fastf1)
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
        print("\n4. Keep ratings current after a race:")
        print("   python run_predictor.py --update-race results.csv --circuit Bahrain --round 2")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
