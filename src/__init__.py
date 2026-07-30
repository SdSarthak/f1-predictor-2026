"""
F1 Predictor - Package Initialization

Public names are resolved lazily so that importing `src` does not pull in
heavy optional dependencies (FastF1, XGBoost) unless they are actually used.
"""

from importlib import import_module
from typing import Any

__version__ = "1.0.0"

_EXPORTS = {
    "F1Predictor": ".predictor",
    "F1DataPipeline": ".data.pipeline",
    "ErgastAPI": ".data.ergast_api",
    "FastF1Client": ".data.fastf1_client",
    "Testing2026Data": ".data.testing_data_2026",
    "FeatureEngineer": ".features.feature_engineering",
    "F1ModelTrainer": ".models.trainer",
    "F1EloRatingSystem": ".models.elo_updater",
    "RaceByRaceUpdater": ".models.race_updater",
    "MonteCarloSimulator": ".simulation.monte_carlo",
    "Regulations2026": ".regulations.rules_2026",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    try:
        module_path = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = import_module(module_path, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return __all__
