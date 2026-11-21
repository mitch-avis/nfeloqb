"""Tests for the weather adjustment a game context applies to quarterback values."""

import math
import pathlib

import pytest

from nfeloqb.DataModels import ModelConfig
from nfeloqb.DataModels.GameContext import GameContext

MODEL_CONFIG = pathlib.Path(__file__).parent.parent / "model_config.json"


def _weather_adj(temp: float | None, wind: float | None) -> float:
    """Return the weather adjustment for one game with the repo's model config."""
    context = GameContext(
        game_id="2030_01_AAA_BBB",
        config=ModelConfig.from_file(str(MODEL_CONFIG)),
        temp=temp,
        wind=wind,
    )
    return context.weather_adj()


@pytest.mark.parametrize(
    ("temp", "wind"),
    [
        (math.nan, math.nan),
        (math.nan, None),
        (None, math.nan),
    ],
)
def test_weather_adj_nan_weather_matches_missing_weather(
    temp: float | None, wind: float | None
) -> None:
    # Arrange
    missing_weather_adj = _weather_adj(temp=None, wind=None)

    # Act
    result = _weather_adj(temp=temp, wind=wind)

    # Assert
    assert result == pytest.approx(missing_weather_adj)


def test_weather_adj_nan_weather_is_milder_than_worst_case() -> None:
    # Arrange
    worst_case_adj = _weather_adj(temp=0, wind=35)

    # Act
    result = _weather_adj(temp=math.nan, wind=math.nan)

    # Assert
    assert result > worst_case_adj
