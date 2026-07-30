"""
Shared naming constants.

Ergast/Jolpica uses snake_case constructor ids (`red_bull`), while the 2026
lookup tables in `config/settings.yaml` and the Elo ratings file use display
names (`Red Bull Racing`).  Everything that needs to cross that boundary goes
through this module so the two vocabularies stay in sync.
"""

from typing import Dict, List, Tuple

CONSTRUCTOR_DISPLAY_NAMES: Dict[str, str] = {
    'red_bull': 'Red Bull Racing',
    'mercedes': 'Mercedes',
    'ferrari': 'Ferrari',
    'mclaren': 'McLaren',
    'aston_martin': 'Aston Martin',
    'alpine': 'Alpine',
    'williams': 'Williams',
    'rb': 'RB',
    'alphatauri': 'RB',
    'sauber': 'Audi (Sauber)',
    'kick_sauber': 'Audi (Sauber)',
    'alfa': 'Audi (Sauber)',
    'audi': 'Audi (Sauber)',
    'haas': 'Haas',
}

# Reverse lookup so display names round-trip back to canonical ids.
CONSTRUCTOR_IDS: Dict[str, str] = {
    'red bull racing': 'red_bull',
    'red bull': 'red_bull',
    'mercedes': 'mercedes',
    'ferrari': 'ferrari',
    'mclaren': 'mclaren',
    'aston martin': 'aston_martin',
    'alpine': 'alpine',
    'williams': 'williams',
    'rb': 'rb',
    'racing bulls': 'rb',
    'audi (sauber)': 'sauber',
    'kick sauber': 'sauber',
    'audi': 'sauber',
    'haas': 'haas',
}

TEAM_ENGINE_2026: Dict[str, str] = {
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

# Confirmed 2026 entry list: (driver_id, code, full name, constructor_id).
GRID_2026: List[Tuple[str, str, str, str]] = [
    ('verstappen', 'VER', 'Max Verstappen', 'red_bull'),
    ('lawson', 'LAW', 'Liam Lawson', 'red_bull'),
    ('russell', 'RUS', 'George Russell', 'mercedes'),
    ('antonelli', 'ANT', 'Kimi Antonelli', 'mercedes'),
    ('leclerc', 'LEC', 'Charles Leclerc', 'ferrari'),
    ('hamilton', 'HAM', 'Lewis Hamilton', 'ferrari'),
    ('norris', 'NOR', 'Lando Norris', 'mclaren'),
    ('piastri', 'PIA', 'Oscar Piastri', 'mclaren'),
    ('alonso', 'ALO', 'Fernando Alonso', 'aston_martin'),
    ('stroll', 'STR', 'Lance Stroll', 'aston_martin'),
    ('gasly', 'GAS', 'Pierre Gasly', 'alpine'),
    ('doohan', 'DOO', 'Jack Doohan', 'alpine'),
    ('albon', 'ALB', 'Alex Albon', 'williams'),
    ('sainz', 'SAI', 'Carlos Sainz', 'williams'),
    ('tsunoda', 'TSU', 'Yuki Tsunoda', 'rb'),
    ('hadjar', 'HAD', 'Isack Hadjar', 'rb'),
    ('hulkenberg', 'HUL', 'Nico Hulkenberg', 'sauber'),
    ('bortoleto', 'BOR', 'Gabriel Bortoleto', 'sauber'),
    ('ocon', 'OCO', 'Esteban Ocon', 'haas'),
    ('bearman', 'BEA', 'Ollie Bearman', 'haas'),
]

# Race distance per circuit, keyed by a substring of the circuit name.
RACE_LAPS: Dict[str, int] = {
    'bahrain': 57,
    'jeddah': 50,
    'melbourne': 58,
    'albert park': 58,
    'australian': 58,
    'suzuka': 53,
    'shanghai': 56,
    'miami': 57,
    'imola': 63,
    'monaco': 78,
    'catalunya': 66,
    'barcelona': 66,
    'gilles villeneuve': 70,
    'canada': 70,
    'red bull ring': 71,
    'austria': 71,
    'silverstone': 52,
    'hungaroring': 70,
    'spa': 44,
    'zandvoort': 72,
    'monza': 53,
    'baku': 51,
    'singapore': 62,
    'americas': 56,
    'austin': 56,
    'rodriguez': 71,
    'mexico': 71,
    'interlagos': 71,
    'brazil': 71,
    'las vegas': 50,
    'losail': 57,
    'qatar': 57,
    'yas marina': 58,
    'abu dhabi': 58,
}

DEFAULT_RACE_LAPS = 57


def constructor_display_name(constructor_id: str) -> str:
    """Map a constructor id (or already-display name) to its display name."""
    if not constructor_id:
        return 'Unknown'
    return CONSTRUCTOR_DISPLAY_NAMES.get(str(constructor_id).lower(), str(constructor_id))


def constructor_id_from_name(name: str) -> str:
    """Map a display name (or already-canonical id) back to a constructor id."""
    if not name:
        return 'unknown'
    key = str(name).lower()
    if key in CONSTRUCTOR_DISPLAY_NAMES:
        return key
    return CONSTRUCTOR_IDS.get(key, key.replace(' ', '_'))


def race_laps_for(circuit_name: str) -> int:
    """Scheduled race distance for a circuit, or the series-typical default."""
    if not circuit_name:
        return DEFAULT_RACE_LAPS

    name = str(circuit_name).lower()
    for key, laps in RACE_LAPS.items():
        if key in name:
            return laps

    return DEFAULT_RACE_LAPS
