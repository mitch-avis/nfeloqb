"""Focused tests for DataLoader game metadata joins."""

import pandas as pd

from nfeloqb.Resources.data_loader import DataLoader


def _build_loader(games: pd.DataFrame) -> DataLoader:
    """Construct a DataLoader instance without triggering live data pulls."""

    loader = object.__new__(DataLoader)
    loader.db = {"games": games}
    loader.games = None
    return loader


def test_add_game_data_uses_game_id_from_games_index() -> None:
    """Join game metadata when the loaded games frame stores game_id in the index."""

    games = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "gameday": ["2025-09-07"],
            "home_team": ["ATL"],
            "away_team": ["CAR"],
            "home_qb_id": ["00-0000001"],
            "home_qb_name": ["Home QB"],
            "away_qb_id": ["00-0000002"],
            "away_qb_name": ["Away QB"],
            "wind": [6.0],
            "temp": [72.0],
        },
        index=pd.Index(["2025_01_CAR_ATL"], name="game_id"),
    )
    stats = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "team": ["ATL", "CAR"],
        }
    )

    loader = _build_loader(games)
    merged = loader.add_game_data(stats)

    assert merged is not None
    assert merged["game_id"].tolist() == ["2025_01_CAR_ATL", "2025_01_CAR_ATL"]
    assert loader.games is not None
    assert loader.games["game_id"].tolist() == ["2025_01_CAR_ATL"]


def test_add_game_data_preserves_existing_game_id_column() -> None:
    """Keep working when game_id is already exposed as a regular column."""

    games = pd.DataFrame(
        {
            "game_id": ["2025_01_CAR_ATL"],
            "season": [2025],
            "week": [1],
            "gameday": ["2025-09-07"],
            "home_team": ["ATL"],
            "away_team": ["CAR"],
            "home_qb_id": ["00-0000001"],
            "home_qb_name": ["Home QB"],
            "away_qb_id": ["00-0000002"],
            "away_qb_name": ["Away QB"],
            "wind": [6.0],
            "temp": [72.0],
        }
    )
    stats = pd.DataFrame(
        {
            "season": [2025],
            "week": [1],
            "team": ["ATL"],
        }
    )

    loader = _build_loader(games)
    merged = loader.add_game_data(stats)

    assert merged is not None
    assert merged.loc[0, "game_id"] == "2025_01_CAR_ATL"