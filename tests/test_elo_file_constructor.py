"""Focused tests for Elo file game ID handling."""

import pandas as pd

from nfeloqb.Resources.elo_file_constructor import EloConstructor


def _build_constructor(new_file: pd.DataFrame, games: pd.DataFrame) -> EloConstructor:
    """Construct an EloConstructor instance without running the full model pipeline."""
    constructor = object.__new__(EloConstructor)
    constructor.new_file = new_file
    constructor.games = games
    return constructor


def test_add_game_id_and_week_tolerates_unresolved_string_ids() -> None:
    """Leave unresolved game IDs missing instead of crashing on pandas string columns."""
    new_file = pd.DataFrame(
        {
            "date": ["2026-09-10"],
            "season": [2026],
            "team1": ["ATL"],
            "team2": ["CAR"],
            "playoff": [pd.NA],
        }
    )
    games = pd.DataFrame(
        {
            "home_team": pd.Series(dtype="object"),
            "away_team": pd.Series(dtype="object"),
            "season": pd.Series(dtype="int64"),
            "game_type": pd.Series(dtype="object"),
            "game_id": pd.Series(dtype="string"),
            "week": pd.Series(dtype="float64"),
        }
    )

    constructor = _build_constructor(new_file, games)

    constructor.add_game_id_and_week()

    assert constructor.new_file is not None
    assert pd.isna(constructor.new_file.loc[0, "game_id"])
