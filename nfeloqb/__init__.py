"""Expose the main modeling entry points for the nfeloqb package."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feature_optimization import (
        optimize_config,
        optimize_config_subsets,
        optimize_config_subsets_with_rand,
        optimize_config_with_rand,
    )
    from .nfeloqb import run
    from .Resources import DataLoader

_EXPORTS: dict[str, tuple[str, str]] = {
    "optimize_config": (".feature_optimization", "optimize_config"),
    "optimize_config_subsets": (".feature_optimization", "optimize_config_subsets"),
    "optimize_config_subsets_with_rand": (
        ".feature_optimization",
        "optimize_config_subsets_with_rand",
    ),
    "optimize_config_with_rand": (".feature_optimization", "optimize_config_with_rand"),
    "run": (".nfeloqb", "run"),
    "DataLoader": (".Resources", "DataLoader"),
}

__all__ = [
    "optimize_config",
    "optimize_config_subsets",
    "optimize_config_subsets_with_rand",
    "optimize_config_with_rand",
    "run",
    "DataLoader",
]


def __getattr__(name: str) -> Any:
    """Lazily load exports so weekly workflows avoid optional optimizer imports."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
