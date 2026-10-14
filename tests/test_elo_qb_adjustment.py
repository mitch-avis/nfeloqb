"""Tests for how the QB adjustment enters the Elo and QB-adjusted Elo models."""

import math

import pandas as pd
import pytest

from nfeloqb.Resources.elo import Elo

BASE = 1505
QB_ADJ_TO_ELO = 3.3


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
