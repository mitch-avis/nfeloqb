import pandas as pd
import polars as pl

from nfeloqb.Resources.airtable_wrapper import _to_pandas


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
