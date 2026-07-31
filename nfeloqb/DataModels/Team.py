"""Define the mutable team state used while the quarterback model iterates games."""

from dataclasses import dataclass, field

from .ModelConfig import ModelConfig


@dataclass
class Team:
    """Object representing a team."""

    # initing meta
    abbr: str
    config: ModelConfig
    # state
    last_game_date_off: str | None = field(default=None, init=False)
    last_game_season_off: int | None = field(default=None, init=False)
    last_game_date_def: str | None = field(default=None, init=False)
    last_game_season_def: int | None = field(default=None, init=False)
    off_value: float = field(init=False)
    def_value: float = field(init=False)
    season_adjs: float = field(default=0, init=False)
    # extra
    params: dict = field(init=False)

    def __post_init__(self):
        """Initialize the team's starting offensive and defensive values."""
        # init the values
        # unpack params from config for convenience
        self.params = self.config.values
        # init the values
        self.off_value = self.params["init_value"]
        self.def_value = 0  # def is a relative measure, so it is initialized to 0

    def update_off_value(self, value: float, qb_adj: float, gameday: str, season: int) -> None:
        """Update the team's offensive value after a game."""
        # update the value
        self.off_value = (
            self.params["team_off_sf"] * value + (1 - self.params["team_off_sf"]) * self.off_value
        )
        # track the cumulative qb adj
        self.season_adjs += qb_adj
        # update the last game date
        self.last_game_date_off = gameday
        self.last_game_season_off = season

    def update_def_value(self, value: float, gameday: str, season: int) -> None:
        """Update the team's defensive value after a game."""
        self.def_value = (
            self.params["team_def_sf"] * value + (1 - self.params["team_def_sf"]) * self.def_value
        )
        # update the last game date
        self.last_game_date_def = gameday
        self.last_game_season_def = season

    def regress_offense(self, qb_val: float, prev_season_league_avg: float) -> None:
        """Handle the offseason regression of a team's offensive value."""
        # normalize the regression coefficients so they are not greater than 1
        total_regression = (
            self.params["team_off_qb_reversion"] + self.params["team_off_league_reversion"]
        )
        if total_regression > 1:
            self.params["team_off_qb_reversion"] = (
                self.params["team_off_qb_reversion"] / total_regression
            )
            self.params["team_off_league_reversion"] = (
                self.params["team_off_league_reversion"] / total_regression
            )
        # regress
        self.off_value = (
            (1 - self.params["team_off_qb_reversion"] - self.params["team_off_league_reversion"])
            * self.off_value
            + self.params["team_off_qb_reversion"] * qb_val
            + self.params["team_off_league_reversion"] * prev_season_league_avg
        )
        # reset the cumulative qb adj
        self.season_adjs = 0

    def regress_defense(self) -> None:
        """Regress the team's defensive value back toward league-average baseline."""
        self.def_value = (1 - self.params["team_def_reversion"]) * self.def_value + self.params[
            "team_def_reversion"
        ] * 0
