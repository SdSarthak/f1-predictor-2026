"""
2026 Pre-Season Testing Data
Bahrain Test 1 & 2 + Barcelona Shakedown Results

This data is used to inform 2026 performance predictions and reduce uncertainty
for the new regulation era.
"""

from typing import Dict, List
import pandas as pd


class Testing2026Data:
    """
    Container for 2026 pre-season testing data.
    Includes Barcelona shakedown and Bahrain official testing.
    """

    def __init__(self):
        # Barcelona Shakedown (Unofficial - focused on reliability)
        self.barcelona_shakedown = {
            'fastest_times': {
                'Ferrari': {'driver': 'Lewis Hamilton', 'time': 76.348, 'laps': 'approx 500'},
                'Mercedes': {'driver': 'George Russell', 'time': 76.448, 'laps': 1137},
                'McLaren': {'driver': 'Lando Norris', 'time': 76.5, 'laps': 'high'},
                'Red Bull Racing': {'driver': 'Max Verstappen', 'time': 77.0, 'laps': 200},
                'Aston Martin': {'driver': 'Fernando Alonso', 'time': 77.5, 'laps': 'moderate'},
                'Alpine': {'driver': 'Pierre Gasly', 'time': 78.0, 'laps': 'moderate'},
                'Williams': {'driver': 'Alex Albon', 'time': 78.5, 'laps': 'moderate'},
                'RB': {'driver': 'Yuki Tsunoda', 'time': 79.0, 'laps': 'moderate'},
                'Audi (Sauber)': {'driver': 'Nico Hulkenberg', 'time': 79.5, 'laps': 'moderate'},
                'Haas': {'driver': 'Esteban Ocon', 'time': 80.0, 'laps': 'moderate'},
            },
            'reliability_notes': {
                'Mercedes': 'Very strong, 1137 laps completed',
                'Ferrari': 'Strong, nearly 1000 laps completed',
                'Red Bull Racing': 'Cautiously impressive start as new PU manufacturer',
                'Honda': 'Smooth running for Aston Martin and RB',
                'Renault': 'Decent reliability for Alpine',
                'Audi': 'New manufacturer, conservative approach',
            }
        }

        # Bahrain Test 1 & 2 (Official Pre-Season Testing)
        self.bahrain_testing = {
            'fastest_times': {
                # Combined best times from both Bahrain tests
                'Ferrari': {'driver': 'Charles Leclerc', 'time': 91.992, 'day': 3, 'test': 1},
                'Mercedes': {'driver': 'Kimi Antonelli', 'time': 92.803, 'day': 2, 'test': 1},
                'McLaren': {'driver': 'Oscar Piastri', 'time': 92.861, 'day': 3, 'test': 1},
                'Red Bull Racing': {'driver': 'Max Verstappen', 'time': 93.109, 'day': 3, 'test': 1},
                'Alpine': {'driver': 'Pierre Gasly', 'time': 93.421, 'day': 3, 'test': 1},
                'Haas': {'driver': 'Ollie Bearman', 'time': 93.487, 'day': 2, 'test': 1},
                'Aston Martin': {'driver': 'Fernando Alonso', 'time': 93.8, 'day': 2, 'test': 1},
                'Williams': {'driver': 'Carlos Sainz', 'time': 94.0, 'day': 3, 'test': 1},
                'RB': {'driver': 'Yuki Tsunoda', 'time': 94.2, 'day': 2, 'test': 1},
                'Audi (Sauber)': {'driver': 'Gabriel Bortoleto', 'time': 94.5, 'day': 3, 'test': 2},
            },
            'total_laps': {
                'Mercedes': 450,
                'Ferrari': 440,
                'McLaren': 435,
                'Red Bull Racing': 420,
                'Aston Martin': 410,
                'Williams': 400,
                'Alpine': 390,
                'RB': 385,
                'Haas': 380,
                'Audi (Sauber)': 370,
            },
            'reliability_incidents': {
                'Red Bull Racing': 'Minor PU sensor issue on Day 1 - new manufacturer teething',
                'Audi (Sauber)': 'Battery deployment calibration issues - new PU expected',
                'Alpine': 'Gearbox issue on Day 2',
                'Williams': 'Minor hydraulic leak',
            }
        }

    def get_australia_gp_performance_rating(self, team: str) -> float:
        """
        Calculate performance rating from Australia GP 2026 (ACTUAL RACE WEEKEND).
        This overrides testing data with real qualifying results.

        Returns:
            Performance rating 0-1 (1.0 = best)
        """
        # Australia GP Qualifying - Season Opener
        australia_quali_times = {
            'Mercedes': 78.518,           # Russell pole
            'Ferrari': 79.327,            # Leclerc P4 (using slower Ferrari)
            'McLaren': 79.380,            # Piastri P5
            'RB': 79.994,                 # Lindblad P8 (using slower RB)
            'Haas': 80.311,               # Bearman P12
            'Alpine': 80.501,             # Gasly P14
            'Audi (Sauber)': 80.303,      # Hulkenberg P11
            'Williams': 80.941,           # Albon P15
            'Aston Martin': 81.969,       # Alonso P17
            'Red Bull Racing': 99.999,    # Verstappen DNF - huge penalty
        }

        if team not in australia_quali_times:
            return 0.5

        fastest_time = min(australia_quali_times.values())
        team_time = australia_quali_times[team]

        # Verstappen crash gets special treatment
        if team_time > 90:
            return 0.3  # Crashed/DNS penalty

        # Convert to rating
        time_delta = team_time - fastest_time
        rating = max(0.3, 1.0 - (time_delta / 4.0))

        return rating

    def get_testing_performance_rating(self, team: str) -> float:
        """
        Calculate overall testing performance rating (0-1 scale).

        Combines:
        - Bahrain lap time (60% weight - most relevant)
        - Barcelona reliability (20% weight)
        - Total laps completed (20% weight)
        """
        # Normalize Bahrain times (fastest = 1.0)
        bahrain_times = self.bahrain_testing['fastest_times']
        if team not in bahrain_times:
            return 0.5  # Default for unknown teams

        fastest_time = min(t['time'] for t in bahrain_times.values())
        team_time = bahrain_times[team]['time']

        # Convert to rating (faster = higher rating)
        time_delta = team_time - fastest_time
        time_rating = max(0.0, 1.0 - (time_delta / 3.0))  # 3 second delta = 0 rating

        # Lap count rating
        total_laps = self.bahrain_testing['total_laps']
        max_laps = max(total_laps.values())
        lap_rating = total_laps.get(team, 300) / max_laps

        # Reliability rating (inverse of incidents)
        reliability_rating = 1.0
        if team in self.bahrain_testing['reliability_incidents']:
            reliability_rating = 0.85  # Minor penalty for issues

        # Weighted combination
        overall_rating = (
            time_rating * 0.6 +
            lap_rating * 0.2 +
            reliability_rating * 0.2
        )

        # NEW: If Australia GP data available, weight it heavily (70% GP, 30% testing)
        # Australia GP is REAL race weekend data, far more valuable than testing
        australia_rating = self.get_australia_gp_performance_rating(team)
        if australia_rating > 0.3:  # If we have valid Australia data
            overall_rating = australia_rating * 0.7 + overall_rating * 0.3

        return min(1.0, max(0.3, overall_rating))

    def get_power_unit_reliability_score(self, engine: str) -> float:
        """
        Get PU reliability based on Australia GP 2026 + testing.
        Australia GP data weighted 70%, testing 30%.

        Args:
            engine: Engine manufacturer name

        Returns:
            Reliability score (0.7-0.98)
        """
        # Testing performance (30% weight)
        testing_reliability = {
            'Mercedes': 0.97,
            'Ferrari': 0.95,
            'Honda': 0.92,
            'Renault': 0.88,
            'Red Bull Powertrains-Ford': 0.83,
            'Audi': 0.78,
        }

        # Australia GP 2026 ACTUAL performance (70% weight)
        australia_reliability = {
            'Mercedes': 0.98,      # Perfect - Russell pole, Antonelli P2
            'Ferrari': 0.94,       # Good - both cars competitive
            'Honda': 0.95,         # Excellent for RB (3 cars all performed)
            'Renault': 0.89,       # Decent - both Alpine cars finished quali
            'Red Bull Powertrains-Ford': 0.75,  # POOR - Verstappen crash/brake lock
            'Audi': 0.82,          # Issues - Bortoleto breakdown, Hulkenberg gremlins
            'Mercedes-Customer': 0.96,  # McLaren strong, Williams had ERS issue
            'Ferrari-Customer': 0.93,   # Haas solid
            'Honda-Customer': 0.70,     # Aston Martin TERRIBLE - Stroll DNS, Alonso P17
        }

        # Get ratings with fallback
        test_score = testing_reliability.get(engine, 0.85)
        aus_score = australia_reliability.get(engine, test_score)

        # Weighted average (Australia GP counts for 70%)
        final_score = aus_score * 0.7 + test_score * 0.3

        return final_score

    def get_team_testing_uncertainty_adjustment(self, team: str) -> float:
        """
        Adjustment factor for prediction uncertainty based on testing.

        Teams that ran well and completed many laps should have lower uncertainty.

        Returns:
            Multiplier for uncertainty (0.7-1.2)
            - Lower = more confident (good testing)
            - Higher = less confident (poor testing)
        """
        total_laps = self.bahrain_testing['total_laps'].get(team, 350)
        max_laps = max(self.bahrain_testing['total_laps'].values())

        # Teams with more laps = lower uncertainty
        lap_factor = total_laps / max_laps

        # Reliability incidents increase uncertainty
        incident_penalty = 1.0
        if team in self.bahrain_testing['reliability_incidents']:
            incident_penalty = 1.15

        # Calculate adjustment (high laps = multiplier closer to 0.7)
        base_adjustment = 1.2 - (lap_factor * 0.5)
        final_adjustment = base_adjustment * incident_penalty

        return min(1.3, max(0.7, final_adjustment))

    def get_testing_summary_dataframe(self) -> pd.DataFrame:
        """
        Get testing data as a pandas DataFrame for analysis.
        """
        data = []

        for team in self.bahrain_testing['fastest_times'].keys():
            bahrain = self.bahrain_testing['fastest_times'][team]

            row = {
                'Team': team,
                'Bahrain_Best_Time': bahrain['time'],
                'Bahrain_Driver': bahrain['driver'],
                'Total_Laps': self.bahrain_testing['total_laps'].get(team, 0),
                'Performance_Rating': self.get_testing_performance_rating(team),
                'Uncertainty_Adjustment': self.get_team_testing_uncertainty_adjustment(team),
                'Had_Issues': team in self.bahrain_testing['reliability_incidents'],
            }

            data.append(row)

        df = pd.DataFrame(data)
        df = df.sort_values('Bahrain_Best_Time')

        return df

    def get_engine_manufacturer_from_team(self, team: str) -> str:
        """Map team name to engine manufacturer."""
        engine_map = {
            'Red Bull Racing': 'Red Bull Powertrains-Ford',
            'Mercedes': 'Mercedes',
            'Ferrari': 'Ferrari',
            'McLaren': 'Mercedes',
            'Aston Martin': 'Honda',
            'Alpine': 'Renault',
            'Williams': 'Mercedes',
            'RB': 'Honda',
            'Audi (Sauber)': 'Audi',
            'Haas': 'Ferrari',
        }
        return engine_map.get(team, 'Unknown')


def get_2026_testing_data() -> Testing2026Data:
    """
    Factory function to get 2026 testing data.
    """
    return Testing2026Data()


if __name__ == "__main__":
    # Example usage
    testing = Testing2026Data()

    print("2026 Pre-Season Testing Summary")
    print("=" * 60)
    print("\nTeam Performance Ratings:")

    df = testing.get_testing_summary_dataframe()
    print(df.to_string(index=False))

    print("\n\nPower Unit Reliability Scores:")
    for engine in ['Mercedes', 'Ferrari', 'Honda', 'Renault', 'Red Bull Powertrains-Ford', 'Audi']:
        score = testing.get_power_unit_reliability_score(engine)
        print(f"  {engine}: {score:.2f}")
