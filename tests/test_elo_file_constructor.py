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


def test_get_next_games_keeps_only_unplayed_games_from_next_week() -> None:
    """Exclude already-played games from the selected next week."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    games = pd.DataFrame(
        {
            "game_id": ["played", "next", "later"],
            "gameday": [
                today - pd.Timedelta(days=1),
                today + pd.Timedelta(days=1),
                today + pd.Timedelta(days=8),
            ],
            "result": [7.0, pd.NA, pd.NA],
            "season": [2026, 2026, 2026],
            "week": [2, 2, 3],
            "home_team": ["ATL", "CAR", "DAL"],
            "away_team": ["NO", "TB", "PHI"],
        }
    )

    constructor = object.__new__(EloConstructor)
    constructor.games = games

    constructor.get_next_games()

    assert constructor.next_games is not None
    assert constructor.next_games["game_id"].tolist() == ["next"]


def test_create_new_file_formats_projected_gameday_as_iso_date() -> None:
    """Projected rows should export dates without time or timezone components."""
    constructor = object.__new__(EloConstructor)
    constructor.original_elo_file = pd.DataFrame(columns=["date"])
    constructor.original_elo_cols = constructor.original_elo_file.columns.to_list()
    constructor.new_file_data = []
    constructor.new_file_games = pd.DataFrame(
        {
            "gameday": [pd.Timestamp("2026-09-20 00:00:00+00:00")],
            "season": [2026],
            "week": [2],
            "home_team": ["BUF"],
            "away_team": ["DET"],
            "home_score": [pd.NA],
            "away_score": [pd.NA],
            "qb1": ["Josh Allen"],
            "qb2": ["Jared Goff"],
            "qb1_value_pre": [1.0],
            "qb2_value_pre": [2.0],
            "qb1_value_post": [1.5],
            "qb2_value_post": [2.5],
            "qb1_adj": [0.1],
            "qb2_adj": [0.2],
            "qb1_game_value": [1.2],
            "qb2_game_value": [2.2],
            "elo1_pre": [1600.0],
            "elo2_pre": [1500.0],
            "elo_prob1": [0.6],
            "elo_prob2": [0.4],
            "elo1_post": [pd.NA],
            "elo2_post": [pd.NA],
            "qbelo1_pre": [1601.0],
            "qbelo2_pre": [1501.0],
            "qbelo_prob1": [0.61],
            "qbelo_prob2": [0.39],
            "qbelo1_post": [pd.NA],
            "qbelo2_post": [pd.NA],
            "location": ["Home"],
            "game_type": ["REG"],
        }
    )

    constructor.create_new_file()

    assert constructor.new_file is not None
    assert constructor.new_file.loc[0, "date"] == "2026-09-20"
