"""Build the metadata export that maps Elo quarterback names to GSIS player records."""

# built in
import json
import os
import pathlib
from typing import Any, cast

import nfelodcm as dcm
import numpy
import pandas as pd

## Temporary direct pull until nfelodcm supports windowed/latest iter pulls ##
ROSTER_DOWNLOAD_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
)


def _rename_columns(df: Any, rename_map: dict[str, str]) -> pd.DataFrame:
    """Return a dataframe copy with selected columns renamed."""
    renamed = pd.DataFrame(df).copy()
    renamed.columns = [rename_map.get(str(column), str(column)) for column in renamed.columns]
    return renamed


class MetaConstructor:
    """Create the metadata dataframe used to map historic Elo names to GSIS identifiers.

    The generated export combines historical Elo quarterback names, nflverse player metadata, and
    curated overrides required by downstream consumers.
    """

    def __init__(
        self,
        # player data from DataLoader
        players: pd.DataFrame,
        # new_file from a constructor that has run construct_elo_file()
        elo_file: pd.DataFrame,
    ):
        """Initialize the metadata constructor with player and Elo inputs."""
        self.players = players
        self.elo_file = elo_file
        # additional
        self.package_loc = pathlib.Path(__file__).parent.parent.parent.resolve()
        with open(
            f"{self.package_loc}/nfeloqb/Manual Data/name_id_repl.json",
            encoding="utf-8",
        ) as replacement_file:
            self.repl = json.load(replacement_file)
        self.missing_draft_data = pd.read_csv(
            f"{self.package_loc}/nfeloqb/Manual Data/missing_draft_data.csv",
            index_col=0,
        )
        self.gen_file()

    def get_538_qbs(self):
        """Return Elo quarterbacks after applying the curated name replacements.

        This keeps the downstream join aligned with the names used by the player feed.
        """
        flat_df = pd.concat(
            [
                _rename_columns(self.elo_file[["qb1"]], {"qb1": "name_id"}),
                _rename_columns(self.elo_file[["qb2"]], {"qb2": "name_id"}),
            ]
        )
        flat_df = flat_df[~pd.isnull(flat_df["name_id"])]
        # change the name id for mapping
        flat_df["name_id"] = cast(Any, flat_df["name_id"]).replace(self.repl["elo_repl"])
        flat_df = flat_df.drop_duplicates()
        # return
        return flat_df

    def get_fastr_qbs(self):
        """Return quarterback rows from the player feed with the required name mappings."""
        # isolate qbs
        qbs = self.players[
            # QBs
            (self.players["position"] == "QB")
            |
            # Or players who used to be qbs and no longer satisfy
            # the position requirement
            (
                numpy.isin(
                    self.players["display_name"],
                    ["Tim Tebow", "Terrelle Pryor", "Taysom Hill", "Kendall Hinton"],
                )
            )
        ].copy()
        # filter to essential fields based on drizzle schema requirements
        qbs = _rename_columns(
            qbs[
                [
                    "gsis_id",
                    "display_name",
                    "football_name",
                    "first_name",
                    "last_name",
                    "short_name",
                    "status",
                    "birth_date",
                    "college_name",
                    "height",
                    "weight",
                    "entry_year",
                    "rookie_year",
                    "draft_number",
                    "draft_round",
                    "draft_club",
                    "headshot",
                ]
            ],
            {
                "college_name": "college",
                "draft_club": "draft_team",
                "headshot": "headshot_url",
            },
        )
        # perform replacement on name
        qbs["display_name"] = qbs["display_name"].replace(self.repl["fastr_repl"])
        qbs = qbs.groupby(["gsis_id"]).head(1)
        # return
        return qbs

    def add_missing_draft_data(self, df):
        """Fill missing draft fields from the curated draft backfill file."""
        # load missing draft data
        missing_draft = pd.read_csv(
            f"{self.package_loc}/nfeloqb/Manual Data/missing_draft_data.csv",
            index_col=0,
        )
        # avoid dupes
        missing_draft = missing_draft.groupby(["player_id"]).head(1)
        # add missing draft data
        df = pd.merge(
            df,
            _rename_columns(
                missing_draft[
                    ["player_id", "rookie_year", "draft_number", "entry_year", "birth_date"]
                ],
                {
                    "player_id": "gsis_id",
                    "rookie_year": "rookie_year_fill",
                    "draft_number": "draft_number_fill",
                    "entry_year": "entry_year_fill",
                    "birth_date": "birth_date_fill",
                },
            ),
            on="gsis_id",
            how="left",
        )
        # fill in missing data
        for col in ["rookie_year", "draft_number", "entry_year", "birth_date"]:
            df[col] = df[col].fillna(df[f"{col}_fill"])
        # drop fill cols
        df = df.drop(
            columns=[
                "rookie_year_fill",
                "draft_number_fill",
                "entry_year_fill",
                "birth_date_fill",
            ]
        )
        # change birth date to dob
        df = _rename_columns(df, {"birth_date": "dob"})
        # return
        return df

    def add_manual_data(self, df):
        """Apply curated manual metadata overrides to the merged dataframe."""
        # load manual data
        # check that its there
        if not os.path.exists(f"{self.package_loc}/nfeloqb/Manual Data/manual_data.csv"):
            return df
        # and that it is not empty
        manual_data = pd.read_csv(
            f"{self.package_loc}/nfeloqb/Manual Data/manual_data.csv", index_col=0
        )
        if len(manual_data) == 0:
            return df
        # prep manual for merge
        fill_cols = [col for col in manual_data.columns if col != "name_id"]
        manual_data = _rename_columns(manual_data, {col: f"{col}_fill" for col in fill_cols})
        # merge
        df = pd.merge(df, manual_data[["name_id"] + fill_cols], on="name_id", how="left")
        # fill in missing data
        for col in fill_cols:
            df[col] = df[col].fillna(df[f"{col}_fill"])
        # drop fill cols
        df = df.drop(columns=fill_cols)
        # return
        return df

    def get_latest_roster_status(self):
        """Pull current-season roster status from nflverse.

        Temporary direct HTTP pull until nfelodcm supports windowed/latest
        iter pulls for the rosters table.
        """
        season, _ = dcm.get_season_state()
        roster = pd.read_csv(ROSTER_DOWNLOAD_URL.format(season=season))
        roster = roster[~pd.isnull(roster["gsis_id"])]
        roster = roster.groupby("gsis_id").tail(1)
        return _rename_columns(roster[["gsis_id", "status"]], {"status": "roster_status"})

    def apply_roster_status(self, df, roster_status):
        """Overwrite status using current-season roster presence.

        Players with a gsis_id not on the roster are marked RET.
        Elo-only historic rows (no gsis_id) keep their existing status.
        """
        df = pd.merge(df, roster_status, on="gsis_id", how="left")
        # on roster -> use roster status
        df["status"] = numpy.where(
            ~pd.isnull(df["roster_status"]),
            df["roster_status"],
            df["status"],
        )
        # has gsis_id but not on roster -> RET
        df["status"] = numpy.where(
            (~pd.isnull(df["gsis_id"])) & (pd.isnull(df["roster_status"])),
            "RET",
            df["status"],
        )
        df = df.drop(columns=["roster_status"])
        return df

    def gen_file(self):
        """Generate and write the final metadata export."""
        # get 538 qbs
        qb_elo = self.get_538_qbs()
        # get fastr qbs
        qbs_fastr = self.get_fastr_qbs()
        # add missing draft data
        qbs_fastr = self.add_missing_draft_data(qbs_fastr)
        # merge on name
        qbs_fastr["name_id"] = qbs_fastr["display_name"]
        df = pd.merge(qb_elo, qbs_fastr, on="name_id", how="outer")
        # note misses
        elo_misses = len(df[df["gsis_id"].isna()])
        fastr_misses = len(df[df["name_id"].isna()])
        duplicate_gsis = len(df[(~pd.isnull(df["gsis_id"])) & (df["gsis_id"].duplicated())])
        if elo_misses > 0:
            print(f"WARN: {elo_misses} elo qbs not found in fastr:")
            if elo_misses > 324:
                print("     This is more than expected! Check the elo file.")
        if fastr_misses > 0:
            print(f"ERROR: {fastr_misses} fastr qbs not found in elo")
            for name in df[df["name_id"].isna()]["display_name"]:
                print(f"     {name}")
        if duplicate_gsis > 0:
            print(f"ERROR: {duplicate_gsis} duplicate gsis ids in meta data")
            for name_id in df[(df["gsis_id"].duplicated()) & (~pd.isnull(df["gsis_id"]))][
                "name_id"
            ]:
                print(f"     {name_id}")
        # overwrite status from current-season roster
        roster_status = self.get_latest_roster_status()
        df = self.apply_roster_status(df, roster_status)
        # add manual data
        df = self.add_manual_data(df)
        df = df.sort_values(by=["name_id"]).reset_index(drop=True)
        # final clean up of edge cases
        df["name_id"] = numpy.where(
            df["gsis_id"] == "00-0035723", "Vincent Testaverde", df["name_id"]
        )
        # save
        df.to_csv(f"{self.package_loc}/Other Data/meta_data.csv")
