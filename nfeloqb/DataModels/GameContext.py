"""Capture per-game context used to apply weather-based quarterback adjustments."""

# Built-in
from dataclasses import dataclass, field

# Packages
import pandas as pd

# Models
from .ModelConfig import ModelConfig

# utilities
from .Utilities import s_curve


def _known(value: float | None) -> float | None:
    """Return the reading, or ``None`` when it is missing as ``None`` or NaN.

    Dome games arrive from the games data with NaN, not ``None``, for wind and temperature.
    """
    if value is None or pd.isna(value):
        return None
    return value


@dataclass
class GameContext:
    """Store the environmental context needed to score a quarterback game."""

    # initing meta
    game_id: str
    config: ModelConfig
    # optional
    temp: float | None = None
    wind: float | None = None
    # post init
    params: dict = field(init=False)

    def __post_init__(self):
        """Unpack the config values for convenient local access."""
        # unpack the config for convenience
        self.params = self.config.values

    def weather_adj(self) -> float:
        """Calculate the combined weather adjustment for wind and temperature."""
        # handle values
        known_wind = _known(self.wind)
        known_temp = _known(self.temp)
        wind = max(0, min(30, known_wind - 5 if known_wind is not None else 0))
        temp = max(0, known_temp if known_temp is not None else 70)
        # calc adjs
        wind_adj = s_curve(self.params["wind_disc_height"], self.params["wind_disc_mp"], wind, "up")
        temp_adj = s_curve(
            self.params["temp_disc_height"], self.params["temp_disc_mp"], temp, "down"
        )
        # calc the adjustment
        return temp_adj + wind_adj
