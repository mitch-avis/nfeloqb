import pandas as pd
import polars as pl

from nfeloqb.Resources import airtable_wrapper
from nfeloqb.Resources.airtable_wrapper import AirtableWrapper, _to_pandas


class _FakeUtcTimestamp:
    year = 2026
    month = 7

    def tz_convert(self, zone: str) -> _FakeUtcTimestamp:
        if zone == "UTC":
            return self
        if zone == "US/Eastern":
            raise KeyError(zone)
        msg = f"Unexpected timezone conversion: {zone}"
        raise AssertionError(msg)

    def isoformat(self) -> str:
        return "2026-07-31T00:00:00+00:00"


class _FakeTimestampFactory:
    @staticmethod
    def utcnow() -> _FakeUtcTimestamp:
        return _FakeUtcTimestamp()

    @staticmethod
    def now(*, tz=None) -> _FakeUtcTimestamp:
        _ = tz
        return _FakeUtcTimestamp()


class _FakeNflReadPy:
    def __init__(self) -> None:
        self.depth_chart_calls: list[list[int]] = []

    def load_depth_charts(self, seasons: list[int]) -> pd.DataFrame:
        self.depth_chart_calls.append(list(seasons))
        return pd.DataFrame(
            {
                "team": ["DEN", "DEN", "KC", "KC"],
                "pos_abb": ["QB", "QB", "QB", "QB"],
                "pos_rank": [1, 2, 1, 4],
                "gsis_id": ["00-0039732", "00-0035264", "00-0033873", "00-0037324"],
                "player_name": [
                    "Bo Nix",
                    "Jarrett Stidham",
                    "Patrick Mahomes",
                    "Chris Oladokun",
                ],
                "dt": ["2026-07-31T09:40:30Z"] * 4,
            }
        )

    def load_players(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gsis_id": ["00-0039732", "00-0033873", "00-0035264", "00-0037324"],
                "draft_pick": [12, 10, 133, 241],
            }
        )


def test_to_pandas_with_polars():
    df_pl = pl.DataFrame({"a": [1, 2]})
    df_pd = _to_pandas(df_pl)
    assert isinstance(df_pd, pd.DataFrame)
    assert df_pd["a"].tolist() == [1, 2]


def test_to_pandas_with_pandas():
    df_pd = pd.DataFrame({"a": [3]})
    result = _to_pandas(df_pd)
    assert result is df_pd


def test_to_pandas_with_none():
    assert _to_pandas(None) is None


def test_pull_current_starters_uses_depth_chart_qb1_without_tzdata(monkeypatch) -> None:
    fake_nfl = _FakeNflReadPy()
    stale_model_df = pd.DataFrame(
        {
            "team": ["DEN", "KC"],
            "player_id": ["00-0035264", "00-0037324"],
            "player_display_name": ["Jarrett Stidham", "Chris Oladokun"],
            "draft_number": [133, 241],
            "gameday": ["2026-01-25", "2026-01-04"],
        }
    )

    monkeypatch.setattr(airtable_wrapper, "nfl", fake_nfl)
    monkeypatch.setattr(airtable_wrapper.pd, "Timestamp", _FakeTimestampFactory)

    wrapper = AirtableWrapper(model_df=stale_model_df, at_config={}, perform_starter_update=False)
    wrapper.pull_current_starters()

    starters = wrapper.starters_df.sort_values("team").reset_index(drop=True)

    assert fake_nfl.depth_chart_calls == [[2026]]
    assert starters["team"].tolist() == ["DEN", "KC"]
    assert starters["player_display_name"].tolist() == ["Bo Nix", "Patrick Mahomes"]
    assert starters["player_id"].tolist() == ["00-0039732", "00-0033873"]
    assert starters["draft_number"].tolist() == [12, 10]
