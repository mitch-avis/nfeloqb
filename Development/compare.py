"""Provide development-only comparison helpers for nfeloqb exports and FiveThirtyEight data."""

import pathlib
from typing import cast

import nfelodcm as dcm
import numpy
import pandas as pd


def _select_with_columns(
    df: pd.DataFrame,
    source_columns: list[str],
    target_columns: list[str],
) -> pd.DataFrame:
    """Return a copy of selected columns with the provided output column names."""
    selected = df[source_columns].copy()
    selected.columns = target_columns
    return cast(pd.DataFrame, selected)


def _suffix_selected_columns(df: pd.DataFrame, columns: list[str], suffix: str) -> pd.DataFrame:
    """Return a copy with the selected column names suffixed."""
    renamed = df.copy()
    renamed.columns = [
        f"{column}{suffix}" if column in columns else column for column in renamed.columns
    ]
    return renamed


def compare_qb_file(ext_file_path: str):
    """Compare an external QB file to the current local QB export."""
    # comparison columns
    comparison_cols = ["value_pre", "value_post", "adj", "game_value"]
    # establish path
    root_loc = pathlib.Path(__file__).parent.parent.resolve()
    # load files
    ext = pd.read_csv(ext_file_path)
    cur = pd.read_csv(f"{root_loc}/qb_elos.csv")
    # constrain to after the nfelo model took over
    ext = cast(pd.DataFrame, ext[ext["season"] >= 2023].copy())
    cur = cast(pd.DataFrame, cur[cur["season"] >= 2023].copy())
    # flatten
    ext = pd.concat(
        [
            _select_with_columns(
                ext,
                [
                    "qb1",
                    "season",
                    "game_id",
                    "qb1_value_pre",
                    "qb1_value_post",
                    "qb1_adj",
                    "qb1_game_value",
                ],
                ["qb", "season", "game_id", "value_pre", "value_post", "adj", "game_value"],
            ),
            _select_with_columns(
                ext,
                [
                    "qb2",
                    "season",
                    "game_id",
                    "qb2_value_pre",
                    "qb2_value_post",
                    "qb2_adj",
                    "qb2_game_value",
                ],
                ["qb", "season", "game_id", "value_pre", "value_post", "adj", "game_value"],
            ),
        ]
    )
    cur = pd.concat(
        [
            _select_with_columns(
                cur,
                [
                    "qb1",
                    "season",
                    "game_id",
                    "qb1_value_pre",
                    "qb1_value_post",
                    "qb1_adj",
                    "qb1_game_value",
                ],
                ["qb", "season", "game_id", "value_pre", "value_post", "adj", "game_value"],
            ),
            _select_with_columns(
                cur,
                [
                    "qb2",
                    "season",
                    "game_id",
                    "qb2_value_pre",
                    "qb2_value_post",
                    "qb2_adj",
                    "qb2_game_value",
                ],
                ["qb", "season", "game_id", "value_pre", "value_post", "adj", "game_value"],
            ),
        ]
    )
    # rename
    ext = _suffix_selected_columns(ext, comparison_cols, "_ext")
    cur = _suffix_selected_columns(cur, comparison_cols, "_cur")
    # merge
    merged = pd.merge(ext, cur, on=["qb", "season", "game_id"], how="left")
    # add start number and season start number
    merged["start_number"] = merged.groupby(["qb"]).cumcount() + 1
    merged["season_start_number"] = merged.groupby(["qb", "season"]).cumcount() + 1
    # calc diffs
    for col in comparison_cols:
        merged[f"{col}_diff"] = merged[f"{col}_ext"] - merged[f"{col}_cur"]
        merged[f"{col}_abs_diff"] = merged[f"{col}_diff"].abs()
    # print some stats
    print(f"Total Records: {len(merged)}")
    print(f"Missing vs external: {len(merged[merged['value_pre_ext'].isna()]) / len(merged)}")
    print(f"MAE: {merged['value_pre_abs_diff'].mean()}")
    print("20 Largest differences in pre-game value:")
    largest_diffs = merged.sort_values(by="value_pre_abs_diff", ascending=False).head(20)
    for _, raw_row in largest_diffs.iterrows():
        row = raw_row.to_dict()
        sign = "+" if row["value_pre_diff"] > 0 else "-"
        print(
            f"{row['qb']}, Start {row['start_number']}: "
            f"{round(row['value_pre_cur'], 2)} "
            f"({sign}{round(row['value_pre_diff'], 2)})"
        )
    # write to csv
    merged.to_csv(f"{root_loc}/Development/qb_file_comparison.csv", index=False)
    # calcualte errors
    merged["cur_to_cur_ae"] = numpy.absolute(merged["value_pre_cur"] - merged["game_value_cur"])
    merged["cur_to_ext_ae"] = numpy.absolute(merged["value_pre_cur"] - merged["game_value_ext"])
    merged["ext_to_cur_ae"] = numpy.absolute(merged["value_pre_ext"] - merged["game_value_cur"])
    merged["ext_to_ext_ae"] = numpy.absolute(merged["value_pre_ext"] - merged["game_value_ext"])
    # calcualte mae
    mae_df = pd.DataFrame(
        [
            {
                "predictor": "cur",
                "cur_mae": merged["cur_to_cur_ae"].mean(),
                "ext_mae": merged["cur_to_ext_ae"].mean(),
            },
            {
                "predictor": "ext",
                "cur_mae": merged["ext_to_cur_ae"].mean(),
                "ext_mae": merged["ext_to_ext_ae"].mean(),
            },
        ]
    )
    # write to csv
    mae_df.to_csv(f"{root_loc}/Development/qb_file_comparison_mae.csv", index=False)


def compare_to_538():
    """Compare nfelo quarterback predictions to FiveThirtyEight's historical values."""
    # get the flattened model data
    # establish path
    root_loc = pathlib.Path(__file__).parent.parent.resolve()
    qbs = pd.read_csv(f"{root_loc}/Other Data/weekly_qb_states.csv")
    # get the 538 data
    db = dcm.load(["qbelo"])
    qbs_538 = db["qbelo"].copy()
    # constraint to period where 538 was active and give
    # buffer past 1999 to allow model to catch up from inits
    qbs_538 = cast(
        pd.DataFrame,
        qbs_538[(qbs_538["season"] >= 2002) & (qbs_538["season"] <= 2022)].copy(),
    )
    # flatten
    qbs_538 = pd.concat(
        [
            _select_with_columns(
                qbs_538,
                ["game_id", "qb1", "qb1_value_pre", "qb1_value_post", "qb1_game_value"],
                ["game_id", "player_name", "value_pre_538", "value_post_538", "game_value_538"],
            ),
            _select_with_columns(
                qbs_538,
                ["game_id", "qb2", "qb2_value_pre", "qb2_value_post", "qb2_game_value"],
                ["game_id", "player_name", "value_pre_538", "value_post_538", "game_value_538"],
            ),
        ]
    )
    # merge
    merged = pd.merge(qbs_538, qbs, on=["game_id", "player_name"], how="left")
    # make adjs to 538 to make comparable
    # translate elo to value
    for col in ["value_pre_538", "value_post_538", "game_value_538"]:
        merged[col] = merged[col] / 3.3
    # add the def adj
    merged["value_pre_538_def_adj"] = merged["value_pre_538"] - merged["opponent_def_value_pre"]
    # add the performance adj
    merged["game_value_538_def_adj"] = merged["game_value_538"] + merged["opponent_def_value_pre"]
    # calc maes
    merged["f38_to_f38_ae"] = numpy.absolute(
        merged["value_pre_538_def_adj"] - merged["game_value_538"]
    )
    merged["f38_to_nfelo_ae"] = numpy.absolute(
        merged["value_pre_538_def_adj"] - merged["value_performance_def_adj"]
    )
    merged["nfelo_to_f38_ae"] = numpy.absolute(
        merged["value_pre_def_adj"] - merged["game_value_538"]
    )
    merged["nfelo_to_nfelo_ae"] = numpy.absolute(
        merged["value_pre_def_adj"] - merged["value_performance_def_adj"]
    )
    # drop any rows with na values
    merged = merged.dropna()
    # calc maes
    mae_df = pd.DataFrame(
        [
            {
                "model": "538",
                "mae_to_538_value": merged["f38_to_f38_ae"].mean(),
                "mae_to_nfelo_value": merged["f38_to_nfelo_ae"].mean(),
            },
            {
                "model": "nfelo",
                "mae_to_538_value": merged["nfelo_to_f38_ae"].mean(),
                "mae_to_nfelo_value": merged["nfelo_to_nfelo_ae"].mean(),
            },
        ]
    )
    # write to csv
    merged.to_csv(f"{root_loc}/Development/qb_file_comparison_538.csv", index=False)
    mae_df.to_csv(f"{root_loc}/Development/mae_comparison.csv", index=False)
