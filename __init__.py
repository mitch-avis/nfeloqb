"""Expose the top-level public API for the nfeloqb package."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Development import compare_qb_file, compare_to_538
    from .nfeloqb import (
        DataLoader,
        optimize_config,
        optimize_config_subsets,
        optimize_config_subsets_with_rand,
        optimize_config_with_rand,
        run,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "compare_qb_file": (".Development", "compare_qb_file"),
    "compare_to_538": (".Development", "compare_to_538"),
    "DataLoader": (".nfeloqb", "DataLoader"),
    "optimize_config": (".nfeloqb", "optimize_config"),
    "optimize_config_subsets": (".nfeloqb", "optimize_config_subsets"),
    "optimize_config_subsets_with_rand": (".nfeloqb", "optimize_config_subsets_with_rand"),
    "optimize_config_with_rand": (".nfeloqb", "optimize_config_with_rand"),
    "run": (".nfeloqb", "run"),
}

__all__ = [
    "compare_qb_file",
    "compare_to_538",
    "DataLoader",
    "optimize_config",
    "optimize_config_subsets",
    "optimize_config_subsets_with_rand",
    "optimize_config_with_rand",
    "run",
]


def __getattr__(name: str) -> Any:
    """Lazily load exports so package import stays usable without optional tools."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
