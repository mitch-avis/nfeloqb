# wrapper for running the pacakge
# import modules
import datetime
import json

# import env
import os
import pathlib

# get games for last played
import nfelodcm as dcm
import pandas as pd
from dotenv import load_dotenv

from .DataModels import ModelConfig

# import resources
from .Resources import (
    AirtableWrapper,
    DataLoader,
    Elo,
    EloConstructor,
    MetaConstructor,
    QBModel,
)

try:
    ENV_PATH = f"{pathlib.Path(__file__).parent.parent.resolve()}/.env"
    load_dotenv(ENV_PATH)
except (FileNotFoundError, OSError):
    # if running as action, these will already be in env
    pass


def run(perform_starter_update=False, model_only=False, force_run=False):
    # load configs and meta
    config = None
    meta = None
    package_folder = pathlib.Path(__file__).parent.parent.resolve()
    config = ModelConfig.from_file(f"{package_folder}/model_config.json")
    with open(f"{package_folder}/package_meta.json", encoding="utf-8") as fp:
        meta = json.load(fp)
    # init AT
    at_wrapper = AirtableWrapper(
        None,
        at_config={
            "base": os.environ.get("AIRTABLE_BASE"),
            "qb_table": os.environ.get("AIRTABLE_QB_TABLE"),
            "starter_table": os.environ.get("AIRTABLE_START_TABLE"),
            "token": os.environ.get("AIRTABLE_TOKEN"),
            "qb_fields": [os.environ.get("AIRTABLE_QB_FIELDS")],
            "dropdown_field": os.environ.get("AIRTABLE_DROPDOWN_ID"),
        },
        perform_starter_update=perform_starter_update,
    )
    # get last starter change
    last_starter_change = at_wrapper.get_last_update()
    last_package_update = meta["last_updated"]
    last_package_week = meta["last_full_week"]
    # get last full week
    last_full_season, last_full_week = dcm.get_season_state()
    last_full_week = f"{last_full_season}_{last_full_week}"
    # see if update is required
    if last_package_update is not None and not force_run:
        # Only apply skip logic when Airtable is enabled
        if not getattr(at_wrapper, "disabled", False):
            if (
                last_starter_change < pd.to_datetime(last_package_update, utc=True)
                and last_full_week == last_package_week
            ):
                return None
    # load data
    data = DataLoader()
    # run model
    print("Running QB model...")
    model = QBModel(data.model_df, config)  # type: ignore
    model.run_model()
    if model_only:
        return model
    # update starters
    at_wrapper.model_df = model.games  # type: ignore
    at_wrapper.update_qb_table()
    at_wrapper.update_qb_options()
    # run elo model
    print("Running Elo model...")
    elo = Elo(data.games, pd.DataFrame(model.data))
    elo.run()
    # construct elo file
    constructor = EloConstructor(data.games, model, at_wrapper, elo, package_folder)
    constructor.construct_elo_file()
    # save flattened qb and team data
    pd.DataFrame(model.data_team).sort_values(
        by=["team", "season", "week"], ascending=[True, True, True]
    ).reset_index(drop=True).to_csv(
        f"{package_folder}/Other Data/defensive_adjustments.csv", index=False
    )
    # save flattened qb records
    pd.DataFrame(model.qb_records).to_csv(
        f"{package_folder}/Other Data/weekly_qb_states.csv", index=False
    )
    # save meta data
    _ = MetaConstructor(players=data.db["players"], elo_file=constructor.new_file)  # type: ignore
    # update the last updated timestamp
    with open(f"{package_folder}/package_meta.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "last_full_week": last_full_week,
            },
            fp,
        )
