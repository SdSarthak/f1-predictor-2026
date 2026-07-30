"""Data module initialization.

`fastf1` is an optional heavyweight dependency, so the FastF1 helpers are
imported lazily; everything else is available eagerly.
"""

from importlib import import_module
from typing import Any

from .ergast_api import ErgastAPI, fetch_ergast_data
from .pipeline import F1DataPipeline, build_pipeline
from .testing_data_2026 import Testing2026Data, get_2026_testing_data

_LAZY = {
    "FastF1Client": ".fastf1_client",
    "fetch_fastf1_data": ".fastf1_client",
}

__all__ = [
    "ErgastAPI",
    "fetch_ergast_data",
    "FastF1Client",
    "fetch_fastf1_data",
    "F1DataPipeline",
    "build_pipeline",
    "Testing2026Data",
    "get_2026_testing_data",
]


def __getattr__(name: str) -> Any:
    try:
        module_path = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value
