"""Tests for how the QB adjustment enters the Elo and QB-adjusted Elo models."""

import math
import pathlib
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from nfeloqb.DataModels import ModelConfig
from nfeloqb.Resources.elo import Elo
from nfeloqb.Resources.elo_file_constructor import EloConstructor
from nfeloqb.Resources.qb_model import QBModel

BASE = 1505
QB_ADJ_TO_ELO = 3.3
MODEL_CONFIG = pathlib.Path(__file__).parent.parent / "model_config.json"
STARTER = ("00-0000001", "Starter QB", 1.0)
BACKUP = ("00-0000002", "Backup QB", 250.0)
RIVAL = ("00-0000003", "Rival QB", 10.0)


def _run_one_game(home_qb_adj: float, away_qb_adj: float, result: float | None) -> pd.Series:
    """Run the Elo model on one neutral-site game between two new teams.

    The model seeds ratings only for teams with a home game, so an unplayed return game
    gives the away team one.
    """
    games = pd.DataFrame(
        {
            "game_id": ["2030_01_AAA_BBB", "2030_02_BBB_AAA"],
            "season": [2030, 2030],
            "week": [1, 2],
            "home_team": ["BBB", "AAA"],
            "away_team": ["AAA", "BBB"],
            "result": [result, None],
            "location": ["Neutral", "Neutral"],
            "home_rest": [7, 7],
            "away_rest": [7, 7],
        }
    )
    qbs = pd.DataFrame(
        {
            "game_id": ["2030_01_AAA_BBB", "2030_01_AAA_BBB"],
            "team": ["BBB", "AAA"],
            "qb_adj": [home_qb_adj, away_qb_adj],
        }
    )
    elo = Elo(games, qbs)
    elo.run()
    assert elo.elo_df is not None
    return elo.elo_df.iloc[0]


def _win_prob(elo_dif: float) -> float:
    return 1 / (10 ** (-elo_dif / 400) + 1)


def test_equal_qbs_cancel_in_the_qb_adjusted_probability() -> None:
    # Act
    record = _run_one_game(home_qb_adj=10, away_qb_adj=10, result=None)

    # Assert
    assert record["qbelo_prob1"] == pytest.approx(0.5)


def test_a_better_away_qb_lowers_the_home_qb_adjusted_probability() -> None:
    # Act
    record = _run_one_game(home_qb_adj=0, away_qb_adj=10, result=None)

    # Assert
    assert record["qbelo_prob1"] == pytest.approx(_win_prob(-10 * QB_ADJ_TO_ELO))
    assert record["qbelo_prob2"] == pytest.approx(1 - record["qbelo_prob1"])


def test_the_plain_elo_ignores_the_qbs() -> None:
    # Act
    record = _run_one_game(home_qb_adj=10, away_qb_adj=0, result=7)

    # Assert
    assert record["elo_prob1"] == pytest.approx(0.5)
    assert record["elo1_post"] == pytest.approx(BASE + 20 * 0.5 * math.log(8))
    assert record["elo2_post"] == pytest.approx(BASE - 20 * 0.5 * math.log(8))


def test_qb_adjusted_ratings_exclude_the_game_qb_adjustment() -> None:
    # Act
    record = _run_one_game(home_qb_adj=10, away_qb_adj=0, result=7)

    # Assert
    assert record["qbelo1_pre"] == pytest.approx(BASE)
    assert record["qbelo2_pre"] == pytest.approx(BASE)
    home_prob = _win_prob(10 * QB_ADJ_TO_ELO)
    assert record["qbelo_prob1"] == pytest.approx(home_prob)
    mov_mult = math.log(8) * (2.2 / (10 * QB_ADJ_TO_ELO * 0.001 + 2.2))
    assert record["qbelo1_post"] == pytest.approx(BASE + 20 * (1 - home_prob) * mov_mult)


def _project(current_elos: dict[str, float], current_qbelos: dict[str, float]) -> pd.Series:
    """Project one upcoming LV-at-home game against the Rams (`LA`) from given ratings."""
    constructor = object.__new__(EloConstructor)
    constructor.new_games = None
    constructor.next_games = pd.DataFrame(
        {
            "home_team": ["LV"],
            "away_team": ["LA"],
            "qb1_adj": [10.0],
            "qb2_adj": [0.0],
            "home_rest": [7],
            "away_rest": [7],
            "location": ["Neutral"],
        }
    )
    constructor.elo = cast(
        Any,
        SimpleNamespace(
            current_elos=current_elos,
            current_qbelos=current_qbelos,
            current_hfa=50.0,
            rest=25,
            b=400,
        ),
    )
    constructor.add_elo_to_new_and_next()
    assert constructor.next_games is not None
    return constructor.next_games.iloc[0]


def test_projections_find_ratings_stored_under_legacy_team_codes() -> None:
    # Arrange
    elos = {"OAK": 1600.0, "LA": 1500.0}
    qbelos = {"OAK": 1550.0, "LA": 1450.0}

    # Act
    row = _project(elos, qbelos)

    # Assert
    assert row["elo1_pre"] == pytest.approx(1600.0)
    assert row["elo2_pre"] == pytest.approx(1500.0)
    assert row["qbelo1_pre"] == pytest.approx(1550.0)
    assert row["qbelo2_pre"] == pytest.approx(1450.0)
    assert row["elo_prob1"] == pytest.approx(_win_prob(100.0))
    assert row["qbelo_prob1"] == pytest.approx(_win_prob(100.0 + 10 * QB_ADJ_TO_ELO))


class _FakeQBModel:
    """Hold team states by code and create missing ones, as the QB model's `get_team` does."""

    def __init__(self, team_off_values: dict[str, float], qb_value: float) -> None:
        self.teams = {code: SimpleNamespace(off_value=v) for code, v in team_off_values.items()}
        self.qb_value = qb_value

    def get_objects(self, row: dict[str, Any]) -> tuple[Any, Any, Any]:
        team = self.teams.setdefault(row["team"], SimpleNamespace(off_value=25.0))
        opponent = self.teams.setdefault(row["opponent"], SimpleNamespace(off_value=25.0))
        return SimpleNamespace(get_value=lambda _team: self.qb_value), team, opponent


def _project_game(
    qb_model: Any,
    starters: dict[str, tuple[str, str, float]],
    home_team: str,
    away_team: str,
) -> pd.Series:
    """Attach the given starters and their QB adjustments to one upcoming 2030 game."""
    constructor = object.__new__(EloConstructor)
    constructor.qb_model = qb_model
    constructor.at_wrapper = cast(
        Any,
        SimpleNamespace(
            pull_current_starters=lambda: None,
            starters_df=pd.DataFrame(
                {
                    "team": list(starters),
                    "player_id": [qb[0] for qb in starters.values()],
                    "player_display_name": [qb[1] for qb in starters.values()],
                    "draft_number": [qb[2] for qb in starters.values()],
                }
            ),
        ),
    )
    constructor.next_games = pd.DataFrame(
        {
            "season": [2030],
            "gameday": ["2030-10-18"],
            "home_team": [home_team],
            "away_team": [away_team],
        }
    )
    constructor.add_starters_and_team_values()
    assert constructor.next_games is not None
    return constructor.next_games.iloc[0]


def _project_starters(qb_model: _FakeQBModel) -> pd.Series:
    """Attach starters and QB adjustments to one upcoming LV-at-home game against the Rams."""
    starters = {"LV": ("00-1", "Home QB", 1.0), "LA": ("00-2", "Away QB", 2.0)}
    return _project_game(qb_model, starters, home_team="LV", away_team="LA")


def test_projected_qb_adjustments_use_teams_stored_under_legacy_codes() -> None:
    # Arrange
    qb_model = _FakeQBModel({"OAK": 60.0, "LAR": 50.0}, qb_value=70.0)

    # Act
    row = _project_starters(qb_model)

    # Assert
    assert row["qb1_adj"] == pytest.approx(10.0)
    assert row["qb2_adj"] == pytest.approx(20.0)
    assert "LV" not in qb_model.teams


def test_projected_qb_adjustments_prefer_a_team_stored_under_the_modern_code() -> None:
    # Arrange
    qb_model = _FakeQBModel({"OAK": 60.0, "LV": 40.0, "LAR": 50.0}, qb_value=70.0)

    # Act
    row = _project_starters(qb_model)

    # Assert
    assert row["qb1_adj"] == pytest.approx(30.0)


def test_projections_rate_an_unknown_team_at_the_mean_of_each_model() -> None:
    # Arrange
    elos = {"LV": 1600.0, "KC": 1400.0}
    qbelos = {"LV": 1560.0, "KC": 1500.0}

    # Act
    row = _project(elos, qbelos)

    # Assert
    assert row["elo2_pre"] == pytest.approx(1500.0)
    assert row["qbelo2_pre"] == pytest.approx(1530.0)


def _qb_model(aaa_starters: list[tuple[str, str, float]]) -> QBModel:
    """Run the QB model over 2030 weeks in which AAA, starting the given QBs, hosts BBB.

    AAA's QBs outplay the team's expectation every week, so AAA builds a season of positive
    QB adjustments that a backup's first start draws on.
    """
    rows = []
    for week, aaa_qb in enumerate(aaa_starters, start=1):
        for team, opponent, (qb_id, qb_name, draft_number), value in (
            ("AAA", "BBB", aaa_qb, 80.0),
            ("BBB", "AAA", RIVAL, 40.0),
        ):
            rows.append(
                {
                    "game_id": f"2030_{week:02d}_BBB_AAA",
                    "season": 2030,
                    "week": week,
                    "gameday": f"2030-09-{week + 10:02d}",
                    "team": team,
                    "opponent": opponent,
                    "player_id": qb_id,
                    "player_display_name": qb_name,
                    "draft_number": draft_number,
                    "player_VALUE": value,
                    "team_VALUE": value,
                    "temp": 70,
                    "wind": 0,
                }
            )
    model = QBModel(pd.DataFrame(rows), ModelConfig.from_file(str(MODEL_CONFIG)))
    model.run_model()
    return model


def _played_week_four(aaa_qb: tuple[str, str, float]) -> dict[str, Any]:
    """Return AAA's QB model record for week 4 when ``aaa_qb`` starts it after three starts."""
    model = _qb_model([STARTER, STARTER, STARTER, aaa_qb])
    return next(r for r in model.data if r["week"] == 4 and r["team"] == "AAA")


@pytest.mark.parametrize(("aaa_side", "qb_col"), [("home", "qb1"), ("away", "qb2")])
@pytest.mark.parametrize("aaa_qb", [BACKUP, STARTER], ids=["backup_first_start", "regular"])
def test_projected_qb_value_matches_the_played_pre_game_value(
    aaa_side: str, qb_col: str, aaa_qb: tuple[str, str, float]
) -> None:
    # Arrange
    played = _played_week_four(aaa_qb)
    model = _qb_model([STARTER, STARTER, STARTER])
    home, away = ("AAA", "BBB") if aaa_side == "home" else ("BBB", "AAA")

    # Act
    row = _project_game(model, {"AAA": aaa_qb, "BBB": RIVAL}, home_team=home, away_team=away)

    # Assert
    assert row[f"{qb_col}_value_pre"] == pytest.approx(played["qb_value_pre"])
    assert row[f"{qb_col}_adj"] == pytest.approx(played["qb_adj"])


def test_a_backups_first_start_projection_adds_the_team_adjustment_share() -> None:
    # Arrange
    model = _qb_model([STARTER, STARTER, STARTER])
    draft_value = model.get_qb(
        {
            "player_id": BACKUP[0],
            "player_display_name": BACKUP[1],
            "draft_number": BACKUP[2],
            "season": 2030,
            "team": "AAA",
            "gameday": "2030-10-18",
        }
    ).current_value
    share = model.teams["AAA"].season_adjs * model.config.values["player_team_adj_allotment_disc"]

    # Act
    row = _project_game(model, {"AAA": BACKUP, "BBB": RIVAL}, home_team="AAA", away_team="BBB")

    # Assert
    assert share > 0
    assert row["qb1_value_pre"] == pytest.approx(draft_value + share)
