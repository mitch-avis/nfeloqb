"""Capture per-game context used to apply weather-based quarterback adjustments."""

# Built-in
from dataclasses import dataclass, field

# Models
from .ModelConfig import ModelConfig

# utilities
from .Utilities import s_curve


@dataclass
class GameContext:
    """Store the environmental context needed to score a quarterback game."""

    # initing meta
    game_id: str
    config: ModelConfig
    # optional
    temp: int | None = None
    wind: int | None = None
    # post init
    params: dict = field(init=False)

    def __post_init__(self):
        """Unpack the config values for convenient local access."""
        # unpack the config for convenience
        self.params = self.config.values

    def weather_adj(self) -> float:
        """Calculate the combined weather adjustment for wind and temperature."""
        # handle values
        wind = max(0, min(30, self.wind - 5 if self.wind is not None else 0))
        temp = max(0, self.temp if self.temp is not None else 70)
        # calc adjs
        wind_adj = s_curve(self.params["wind_disc_height"], self.params["wind_disc_mp"], wind, "up")
        temp_adj = s_curve(
            self.params["temp_disc_height"], self.params["temp_disc_mp"], temp, "down"
        )
        # calc the adjustment
        return temp_adj + wind_adj
