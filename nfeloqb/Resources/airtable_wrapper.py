import json
import math
import os
import pathlib
import time
from typing import Any, cast
from urllib.error import HTTPError, URLError

import numpy
import pandas as pd
import requests

try:
    import nflreadpy as nfl
except ImportError:
    nfl = None


def _to_pandas(df: Any) -> pd.DataFrame | None:
    if df is None:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    to_pandas = getattr(df, "to_pandas", None)
    if callable(to_pandas):
        try:
            converted = to_pandas()
            return (
                converted
                if isinstance(converted, pd.DataFrame)
                else pd.DataFrame(cast(Any, converted))
            )
        except ModuleNotFoundError as exc:
            if exc.name == "pyarrow":
                to_dicts = getattr(df, "to_dicts", None)
                if callable(to_dicts):
                    return pd.DataFrame(cast(Any, to_dicts()))
            raise
    return pd.DataFrame(cast(Any, df))


class AirtableWrapper:
    # This class is handles IO for an airtable base that stores
    # starters for the current week
    def __init__(self, model_df, at_config, perform_starter_update=True):
        # df of qbs and their meta data
        self.model_df = model_df
        # config for airtable including token, ids, etc
        self.at_config = at_config

        # NEW: disabled mode for local runs or missing creds
        disable_flag = os.getenv("NFELOQB_DISABLE_AIRTABLE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        required = ["base", "qb_table", "starter_table", "token"]
        missing = any(not self.at_config.get(k) for k in required)
        self.disabled = disable_flag or missing

        # Safe defaults for attributes used downstream
        self.existing_qbs = []
        self.existing_qb_options = []
        self.existing_starters = {}
        self.starters_df = pd.DataFrame(
            columns=[
                "team",
                "player_id",
                "player_display_name",
                "draft_number",
                "last_updated",
            ]
        )
        self.all_qbs = None
        self.qb_options = []
        self.perform_starter_update = perform_starter_update

        if not self.disabled:
            # unpack config only if enabled
            self.base = self.at_config["base"]
            self.qb_table = self.at_config["qb_table"]
            self.starter_table = self.at_config["starter_table"]
            self.token = self.at_config["token"]
            self.qb_fields = self.at_config["qb_fields"]
            self.dropdown_field_id = self.at_config["dropdown_field"]
            self.base_headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        else:
            # placeholders to avoid attribute errors
            self.base = self.qb_table = self.starter_table = self.token = None
            self.qb_fields = []
            self.dropdown_field_id = None
            self.base_headers = {}

    # api wrapper functions
    def make_post_request(self, base, table, headers, data):
        # used for creating new records
        # rate limiting
        time.sleep(1 / 4)
        # formulate url
        url = f"https://api.airtable.com/v0/{base}/{table}"
        requests.post(url, headers=headers, data=json.dumps(data), timeout=10)

    def make_patch_request(self, base, table, headers, data):
        # Used for updating existing records
        # rate limiting
        time.sleep(1 / 4)
        # formulate url
        url = f"https://api.airtable.com/v0/{base}/{table}"
        resp = requests.patch(url, headers=headers, data=json.dumps(data), timeout=10)
        if resp.status_code != 200:
            print(f"Error on patch! -- {resp.status_code} -- {resp.content}")

    def make_get_request(self, base, table, headers, params):
        # used to for getting records
        # rate limiting
        time.sleep(1 / 4)
        # formulate url
        url = f"https://api.airtable.com/v0/{base}/{table}"
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        return resp

    def make_delete_request(self, base, table, headers, params):
        # used for deleting records
        # rate limiting
        time.sleep(1 / 4)
        # formulate url
        url = f"https://api.airtable.com/v0/{base}/{table}"
        requests.delete(url, headers=headers, params=params, timeout=10)

    def make_meta_request(self, base, headers):
        # request schema of base
        # rate limiting
        time.sleep(1 / 4)
        # formulate url
        url = f"https://api.airtable.com/v0/meta/bases/{base}/tables"
        resp = requests.get(url, headers=headers, timeout=10)
        return resp.json()

    def make_paginated_get(self, base, table, headers, params):
        if self.disabled:
            return []
        # make first request
        all_records = []
        resp = self.make_get_request(base, table, headers, params)
        records = resp.json()
        # add records to container for initial pull
        for record in records["records"]:
            all_records.append(record)
        # init var loops
        if "offset" in records.keys():
            offset = records["offset"]
            loops = 0
        else:
            offset = None  # if no offset, no need to paginate
            loops = 0
        # loop
        while offset is not None and loops < 50:
            params["offset"] = offset
            resp = self.make_get_request(base, table, headers, params)
            records = resp.json()
            # add records to container for initial pull
            for record in records["records"]:
                all_records.append(record)
            # update var loops
            if "offset" in records.keys():
                offset = records["offset"]
                loops += 1
            else:
                offset = None  # if no offset, no need to paginate
                loops += 1
        # return data
        return all_records

    def data_format(self, datapoint):
        # translates a NaN to None for airtable
        if pd.isnull(datapoint):
            return None
        else:
            return datapoint

    def write_chunk(self, base, table, df):
        # write chunk to airtable
        # container for data to write to airtable
        data = {"records": [], "typecast": True}
        # get table cols
        table_cols = df.columns.values.tolist()
        # iterate through chunk and add to data
        for _, row in df.iterrows():
            record = {"fields": {}}
            for col in table_cols:
                record["fields"][col] = self.data_format(row[col])
            # append to date
            data["records"].append(record)
        # write to table
        self.make_post_request(base=base, table=table, headers=self.base_headers, data=data)

    # write chunk to airtable
    def update_chunk(self, base, table, df, id_col):
        # container for data to write to airtable
        data = {"records": [], "typecast": True}
        # get table cols
        table_cols = df.columns.values.tolist()
        # iterate through chunk and add to data
        for _, row in df.iterrows():
            record = {"id": row[id_col], "fields": {}}
            for col in table_cols:
                if col == id_col:
                    pass
                else:
                    record["fields"][col] = self.data_format(row[col])
            # append to date
            data["records"].append(record)
        # write to table
        self.make_patch_request(base=base, table=table, headers=self.base_headers, data=data)

    # perform upsert to airtable
    def upsert_chunk(self, base, table, df, upsertFields, key):
        # container for data to write to airtable
        data = {"records": [], "performUpsert": {"fieldsToMergeOn": upsertFields}}
        # get table cols
        table_cols = df.columns.values.tolist()
        # control for missing fields
        for field in upsertFields:
            if field not in table_cols:
                print(f"     {field} is not included in data. Upsert will fail...")
        # iterate through chunk and add to data
        for _, row in df.iterrows():
            record = {"fields": {}}
            for col in table_cols:
                record["fields"][col] = self.data_format(row[col])
            # append to date
            data["records"].append(record)
        # write to table
        self.make_patch_request(
            base=base,
            table=table,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            data=data,
        )

    # break df into chunks of 10 and write to airtable
    def write_table(self, base, table, df):
        # break df into chunks of 10
        # determine size of df
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        # split
        df_chunks = numpy.array_split(df, chunks_needed)
        # write
        for chunk in df_chunks:
            # turn chunk into record
            self.write_chunk(base, table, chunk)

    # break df into chunks of 10 and write to airtable
    def update_table(self, base, table, df, id_col):
        # break df into chunks of 10
        # determine size of df
        df_len = len(df)
        chunks_needed = math.ceil(df_len / 10)
        # split
        df_chunks = numpy.array_split(df, chunks_needed)
        # write
        for chunk in df_chunks:
            # turn chunk into record
            self.update_chunk(base, table, chunk, id_col)

    # fucntional abstractions for wrapper
    def get_existing_qbs(self):
        if self.disabled:
            self.existing_qbs = []
            return
        # gets existing QBs from airtable
        # get existing qbs
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.qb_table,
            headers=self.base_headers,
            params={
                # 'fields' : self.qb_fields
            },
        )
        # container for qbs
        qbs = []
        # iterate through qb response and add to container
        for qb in qbs_resp:
            qbs.append(qb["fields"]["player_id"])
        # return
        self.existing_qbs = qbs

    def get_qb_options(self):
        if self.disabled:
            self.qb_options = []
            return
        # gets a list of QBs that are options in the drop down
        # get base schema
        base_schema = self.make_meta_request(
            base=self.base,
            headers=self.base_headers,
        )
        # parse
        options = []
        for table in base_schema["tables"]:
            if table["id"] == self.starter_table:
                for field in table["fields"]:
                    if field["id"] == self.dropdown_field_id:
                        for option in field["options"]["choices"]:
                            options.append(option["name"])
        # return
        self.qb_options = options

    def get_starters(self):
        if self.disabled:
            self.existing_starters = {}
            return
        # gets existing QBs from airtable
        # get existing qbs
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                # 'fields' : self.qb_fields
            },
        )
        # structure for existing starters, which has a key of the team
        # and values of record id and qb_id
        existing_starters = {}
        for record in qbs_resp:
            existing_starters[record["fields"]["team"]] = {
                "record_id": record["id"],
                "qb_id": record["fields"]["qb_id"],
            }
        # write
        self.existing_starters = existing_starters

    def write_qbs(self, qbs_to_write):
        # write a df containing qb meta to the qb db in airtable
        self.write_table(base=self.base, table=self.qb_table, df=qbs_to_write)

    def write_qb_options(self, qb_options_to_write):
        # to update an option to the dropdown, you need to create a record
        # with typecase set to true
        # to do this, loop through new options. On the first, create a dummary record
        # on subsequents records, upsert that record
        # on the final, delete the dummy record
        # container for dummy record id
        dummy_id = None
        for index, value in enumerate(qb_options_to_write):
            # create record structure
            data = {
                "records": [{"fields": {"team": "DUMMY", "qb_id": value}}],
                "typecast": True,
            }
            if index == 0:
                # if first record, create the dummy
                self.make_post_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data,
                )
                # retrieve record to get id
                resp = self.make_get_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={"filterByFormula": 'team = "DUMMY"'},
                )
                resp = resp.json()
                dummy_id = resp["records"][0]["id"]
            else:
                # update record with dummy id
                data["records"][0]["id"] = dummy_id
                # make a patch request
                self.make_patch_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    data=data,
                )
            # if last record, delete dummy
            if index == len(qb_options_to_write) - 1:
                self.make_delete_request(
                    base=self.base,
                    table=self.starter_table,
                    headers=self.base_headers,
                    params={"records[]": dummy_id},
                )

    # model functions
    def get_qbs(self):
        # gets a unique set of QBs from the data file
        # note, this only stores QBs that have made a start
        qbs = self.model_df.copy()
        # get most recent
        qbs = qbs.sort_values(by=["gameday"], ascending=[False]).reset_index(drop=True)
        # add a field that combines id and display name
        qbs["qb_id"] = qbs["player_display_name"] + " - " + qbs["player_id"]
        qbs = (
            qbs[
                [
                    "qb_id",
                    "player_id",
                    "player_display_name",
                    "start_number",
                    "rookie_year",
                    "entry_year",
                    "draft_number",
                ]
            ]
            .groupby(["player_id"])
            .head(1)
        )
        # return
        self.all_qbs = qbs

    def get_last_starter(self):
        # for each team, determines last starter, which is assumed
        # to be the starter for the next week
        starters = self.model_df.copy()
        starters = starters.sort_values(by=["gameday"], ascending=[False]).reset_index(drop=True)
        # add a field that combines id and display name
        starters["qb_id"] = starters["player_display_name"] + " - " + starters["player_id"]
        starters = (
            starters[
                [
                    "team",
                    "qb_id",
                ]
            ]
            .groupby(["team"])
            .head(1)
        )
        # return
        return starters

    # actual functions that get called
    def update_qb_table(self):
        if self.disabled:
            print("Airtable disabled: skipping QB table update.")
            return
        # checks qbs in airtable against qbs in data
        # updates the delta
        print("Updating QB table...")
        # get existing qbs
        self.get_existing_qbs()
        # get qbs from data
        self.get_qbs()
        # get delta
        delta = self.all_qbs[  # type: ignore
            ~numpy.isin(self.all_qbs["player_id"], self.existing_qbs)  # type: ignore
        ].copy()
        # determine write
        if len(delta) > 0:
            print(f"     Found {len(delta)} new QBs")
            # write
            self.write_qbs(delta)
            # update existing qbs so its accurate
            for qb in delta["player_id"].unique().tolist():
                self.existing_qbs.append(qb)
        else:
            print("     No new QBs needed")

    def update_qb_options(self):
        if self.disabled:
            print("Airtable disabled: skipping QB options update.")
            return
        # updates the QB option dropdown to reflect QBs in the
        # database
        print("Updating QB options...")
        # update existing options
        self.get_qb_options()
        # determine all values that should be in dropdown
        delta = self.all_qbs[  # type: ignore
            ~numpy.isin(self.all_qbs["qb_id"], self.qb_options)  # type: ignore
        ].copy()
        # determine write
        if len(delta) > 0:
            print(f"     Found {len(delta)} new QB options")
            # write
            self.write_qb_options(delta["qb_id"].unique().tolist())
        else:
            print("     No new QB options needed")

    def update_starters(self):
        if self.disabled or not self.perform_starter_update:
            return
        # reads the starter table in airtable and determines
        # if any starters are different from the previous week
        print("Updating starters...")
        # get last week's starters from AT
        self.get_starters()
        existing_starters = self.existing_starters
        # get this weeks starters from data
        this_weeks_starters = self.get_last_starter()
        # structure for holding updates
        writes = []
        updates = []
        # loop through teams
        for _, row in this_weeks_starters.iterrows():
            # get team
            team = row["team"]
            if team in existing_starters:
                # if team is in the AT table (it should be) check starter
                if existing_starters[team]["qb_id"] != row["qb_id"]:
                    # if starter is not match, create update rec
                    updates.append(
                        {
                            "id": existing_starters[team]["record_id"],
                            "qb_id": row["qb_id"],
                            # airtable automations dont trigger on API update, so
                            # zero out the fields so it's obvious they need to be updated
                            "start_number": numpy.nan,
                            "rookie_year": numpy.nan,
                            "entry_year": numpy.nan,
                            "draft_number": numpy.nan,
                            "player_display_name": numpy.nan,
                            "player_id": numpy.nan,
                        }
                    )
            else:
                # if team is not in the AT table, create write rec
                writes.append({"team": team, "qb_id": row["qb_id"]})
        # write if necessary
        if len(writes) > 0:
            print(f"     Found {len(writes)} new teams")
            self.write_table(base=self.base, table=self.starter_table, df=pd.DataFrame(writes))
        # update if necessary
        if len(updates) > 0:
            print(f"     Found {len(updates)} updated starters")
            self.update_table(
                base=self.base,
                table=self.starter_table,
                df=pd.DataFrame(updates),
                id_col="id",
            )

    def pull_current_starters(self):
        if self.disabled:
            # Prefer nflverse depth charts (QB1) when Airtable is disabled
            if nfl is not None:
                try:
                    # Determine current season
                    now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
                    now_et = now_utc.tz_convert("US/Eastern")

                    # NFL season spans fall->winter; in Jan/Feb the season year is typically
                    # the prior year.
                    # Use a pragmatic cutoff so we request a year that exists in nflverse datasets.
                    primary_season = now_et.year if now_et.month >= 7 else (now_et.year - 1)
                    candidate_seasons = [primary_season, primary_season - 1]

                    # Load depth charts for the best-available season
                    depth_charts = None
                    last_err: Exception | None = None
                    for season in candidate_seasons:
                        try:
                            depth_charts = _to_pandas(nfl.load_depth_charts([season]))
                            if depth_charts is not None and len(depth_charts) > 0:
                                last_err = None
                                break
                        except (HTTPError, URLError, OSError, ValueError) as e:
                            last_err = e
                            continue

                    if depth_charts is None or len(depth_charts) == 0:
                        if last_err is not None:
                            raise RuntimeError("Depth charts unavailable") from last_err
                        raise RuntimeError("No depth charts returned")

                    # Manually update starting quarterbacks for this week
                    swap_csv_path = (
                        pathlib.Path(__file__).parent.parent / "Manual Data" / "manual_qb_swaps.csv"
                    )
                    if swap_csv_path.exists():
                        manual_swaps = pd.read_csv(swap_csv_path)
                        # Filter out any rows with missing team or pos_rank
                        manual_swaps = manual_swaps.dropna(subset=["team"])

                        # --- Manual depth chart fix for teams in manual_swaps ---
                        for _, swap_row in manual_swaps.iterrows():
                            team = swap_row["team"]
                            # Default to pos_rank=2 if not specified (backward compatibility)
                            target_pos_rank = int(swap_row.get("starting_pos_rank", 2))

                            # Get all QBs for this team
                            team_qbs = depth_charts[
                                (depth_charts["team"] == team) & (depth_charts["pos_abb"] == "QB")
                            ]
                            if not team_qbs.empty:
                                # Find the most recent dt
                                latest_dt = team_qbs["dt"].max()
                                team_qbs_latest = team_qbs[team_qbs["dt"] == latest_dt]

                                # Check if target pos_rank exists
                                if target_pos_rank in team_qbs_latest["pos_rank"].values:
                                    # Get indices for pos_rank 1 and target_pos_rank
                                    idx1 = team_qbs_latest[team_qbs_latest["pos_rank"] == 1].index
                                    idx_target = team_qbs_latest[
                                        team_qbs_latest["pos_rank"] == target_pos_rank
                                    ].index

                                    # Swap pos_rank values
                                    depth_charts.loc[idx1, "pos_rank"] = target_pos_rank
                                    depth_charts.loc[idx_target, "pos_rank"] = 1

                    # Filter to QB1 only
                    quarterbacks = depth_charts[
                        (depth_charts["pos_abb"] == "QB") & (depth_charts["pos_rank"] == 1)
                    ]

                    # Pick the latest entry per team by week/date if present
                    sort_cols = [col for col in ["dt", "team"] if col in quarterbacks.columns]
                    if sort_cols:
                        quarterbacks = quarterbacks.sort_values(by=sort_cols, ascending=True)
                    quarterbacks = quarterbacks.groupby("team", as_index=False).tail(1)

                    # Identify player id and name columns (nflverse uses gsis_id + full_name)
                    pid_col = "gsis_id" if "gsis_id" in quarterbacks.columns else None
                    name_col = "player_name" if "player_name" in quarterbacks.columns else None
                    if pid_col is None or name_col is None:
                        raise RuntimeError("Depth charts missing player id/name columns")

                    starters = quarterbacks[["team", pid_col, name_col]].rename(
                        columns={pid_col: "player_id", name_col: "player_display_name"}
                    )

                    # Attach draft_number from model_df if known
                    if self.model_df is not None and len(self.model_df):
                        drafts = self.model_df[["player_id", "draft_number"]].drop_duplicates(
                            "player_id"
                        )
                        starters = starters.merge(drafts, on="player_id", how="left")
                    else:
                        starters["draft_number"] = numpy.nan

                    # Fill missing draft_number from nflverse players if possible
                    if starters["draft_number"].isna().any():
                        try:
                            players = _to_pandas(nfl.load_players())
                            if players is not None and len(players):
                                p_pid = (
                                    "gsis_id"
                                    if "gsis_id" in players.columns
                                    else ("player_id" if "player_id" in players.columns else None)
                                )
                                draft_col = (
                                    "draft_pick"
                                    if "draft_pick" in players.columns
                                    else (
                                        "draft_number"
                                        if "draft_number" in players.columns
                                        else None
                                    )
                                )
                                name_col2 = (
                                    "full_name"
                                    if "full_name" in players.columns
                                    else (
                                        "display_name"
                                        if "display_name" in players.columns
                                        else None
                                    )
                                )
                                if p_pid and draft_col:
                                    starters = starters.merge(
                                        players[
                                            [p_pid, draft_col] + ([name_col2] if name_col2 else [])
                                        ].rename(
                                            columns={
                                                p_pid: "player_id",
                                                draft_col: "draft_number",
                                            }
                                        ),
                                        on="player_id",
                                        how="left",
                                        suffixes=("", "_players"),
                                    )
                                    # Prefer existing draft_number, fallback to players
                                    starters["draft_number"] = starters["draft_number"].fillna(
                                        starters["draft_number_players"]
                                    )
                                    starters = starters.drop(
                                        columns=[
                                            c for c in starters.columns if c.endswith("_players")
                                        ]
                                    )
                        except KeyError, AttributeError, ValueError:
                            pass

                    # Build qb_id and last_updated
                    starters["qb_id"] = (
                        starters["player_display_name"].astype(str)
                        + " - "
                        + starters["player_id"].astype(str)
                    )
                    starters["last_updated"] = now_utc.isoformat()

                    # Store with the schema EloConstructor expects
                    self.starters_df = starters[
                        [
                            "team",
                            "player_id",
                            "player_display_name",
                            "draft_number",
                            "last_updated",
                            "qb_id",
                        ]
                    ]
                    return
                except (
                    KeyError,
                    AttributeError,
                    ValueError,
                    RuntimeError,
                    HTTPError,
                    URLError,
                    OSError,
                ):
                    # fall back to "last starter" if depth charts fail
                    pass

            # Fallback: derive from most recent games (last starter)
            if self.model_df is None or len(self.model_df) == 0:
                self.starters_df = pd.DataFrame(
                    columns=[
                        "team",
                        "player_id",
                        "player_display_name",
                        "draft_number",
                        "last_updated",
                        "qb_id",
                    ]
                )
                return

            latest = (
                self.model_df.copy()
                .sort_values(by=["gameday"], ascending=[False])
                .reset_index(drop=True)
            )
            latest["qb_id"] = latest["player_display_name"] + " - " + latest["player_id"]
            starters = (
                latest[
                    [
                        "team",
                        "qb_id",
                        "player_id",
                        "player_display_name",
                        "draft_number",
                    ]
                ]
                .groupby("team", as_index=False)
                .head(1)
            )
            now_utc = pd.Timestamp.utcnow().tz_convert("UTC").isoformat()
            starters["last_updated"] = now_utc
            self.starters_df = starters[
                [
                    "team",
                    "player_id",
                    "player_display_name",
                    "draft_number",
                    "last_updated",
                    "qb_id",
                ]
            ]
            return

        # pulls the current starters from the airtable
        # and stores as a DF for the elo constructor
        qbs_resp = self.make_paginated_get(
            base=self.base,
            table=self.starter_table,
            headers=self.base_headers,
            params={
                # 'fields' : self.qb_fields
            },
        )
        # structure for existing starters, which has a key of the team
        # and values of record id and qb_id
        starters_data = []
        for record in qbs_resp:
            # control for missing
            for field in [
                "team",
                "player_id",
                "player_display_name",
                "draft_number",
                "last_updated",
            ]:
                if field not in record["fields"]:
                    record["fields"][field] = numpy.nan
            starters_data.append(
                {
                    "team": record["fields"]["team"],
                    "player_id": record["fields"]["player_id"],
                    "player_display_name": record["fields"]["player_display_name"],
                    "draft_number": record["fields"]["draft_number"],
                    "last_updated": record["fields"]["last_updated"],
                }
            )
        # write
        self.starters_df = pd.DataFrame(starters_data)

    def get_last_update(self):
        """Returns the timestamp when the starters table was last updated"""
        if self.disabled:
            # Return 'now' so we don't skip the build locally
            return pd.Timestamp.utcnow().tz_convert("UTC")
        # get the starters
        self.pull_current_starters()
        starters = self.starters_df.copy()
        # ensure time format
        starters["last_updated"] = pd.to_datetime(
            starters["last_updated"], errors="coerce", utc=True
        )
        # sort
        starters = starters.sort_values(by=["last_updated"], ascending=[False]).reset_index(
            drop=True
        )
        # return most recent
        return starters.iloc[0]["last_updated"]
