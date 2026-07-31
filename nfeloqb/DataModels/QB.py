"""Define the mutable quarterback state tracked by the model across games and seasons."""

# Built-in
import math
from dataclasses import dataclass, field

# Models
from .ModelConfig import ModelConfig
from .Team import Team

# utilities
from .Utilities import prog_disc, s_curve


@dataclass
class QB:
    """Store the quarterback state used by the model across games and seasons."""

    # initing meta
    player_id: str
    player_name: str
    config: ModelConfig
    draft_number: int | None
    inital_league_avg: float
    inital_team_avg: float
    first_game_date: str
    first_game_season: int
    # state
    current_value: float = field(init=False)
    current_variance: float = field(init=False)
    rolling_value: float = field(init=False)
    starts: int = field(init=False)
    season_starts: int = field(init=False)
    season_team_adjs_alotted: float = field(init=False)
    season_team_adjs_received: float = field(init=False)
    season_player_adjs_received: float = field(init=False)
    last_game_date: str | None = field(init=False)
    last_game_season: int | None = field(init=False)
    last_game_team: str | None = field(init=False)
    # extra
    params: dict = field(init=False)

    def __post_init__(self):
        """Initialize quarterback state from the active model configuration."""
        # unpack the config for convenience
        self.params = self.config.values
        # calc the iniitial current value using the draft model
        self.current_value = min(
            # value over team's previous average based on draft number
            (
                self.params["rookie_draft_intercept"]
                + (
                    self.params["rookie_draft_slope"]
                    * math.log(
                        self.draft_number
                        if self.draft_number is not None
                        else self.params["rookie_undrafted_draft_number"]
                    )
                )
            )
            +
            # team value is regressed to the league average
            (
                ((1 - self.params["rookie_league_reg"]) * self.inital_team_avg)
                + (self.params["rookie_league_reg"] * self.inital_league_avg)
            )
            * (1 + self.params["rookie_league_cap"]),
            # value is capped at a discount of the league average
            ((1 + self.params["rookie_league_cap"]) * self.inital_league_avg),
        )
        # set other initial values
        self.current_variance = 1000
        self.rolling_value = self.current_value
        self.starts = 0
        self.season_starts = 0
        self.season_team_adjs_alotted = 0
        self.season_team_adjs_received = 0
        self.season_player_adjs_received = 0
        self.last_game_date = None
        self.last_game_season = None
        self.last_game_team = None

    def as_record(self) -> dict:
        """Return the quarterback state as a dictionary."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "current_value": self.current_value,
            "current_variance": self.current_variance,
            "rolling_value": self.rolling_value,
            "starts": self.starts,
            "season_starts": self.season_starts,
            "season_team_adjs_alotted": self.season_team_adjs_alotted,
            "season_team_adjs_received": self.season_team_adjs_received,
            "season_player_adjs_received": self.season_player_adjs_received,
            "last_game_date": self.last_game_date,
            "last_game_season": self.last_game_season,
            "last_game_team": self.last_game_team,
        }

    def get_value(
        self,
        team_state: Team,
    ) -> float:
        """Return the quarterback value after applying any team-level adjustment."""
        # determine if this QB is a backup and should be adjusted
        if self.season_starts == 0 and team_state.season_adjs != 0:
            # get the cumulative adjustmensts made to QBs on the team
            team_adjs = team_state.season_adjs
            # backup will be worse than the team's average and starter
            value_vs_team = self.current_value - team_state.off_value
            if value_vs_team < -10:
                # determine how many of those adjs have been received by this QB
                # with the current implementation, keeping track of adjs is not necessary because
                # a QB only receives them on their first start, however, this will be left in
                # the data model in case it is needed for future use
                player_adjs = self.season_player_adjs_received + self.season_team_adjs_alotted
                net_adjs = team_adjs - player_adjs
                # update the current value based on net adjs
                other_qb_adj = net_adjs * self.params["player_team_adj_allotment_disc"]
                self.current_value += other_qb_adj
                # update the allotted amount to this QB
                self.season_team_adjs_alotted += net_adjs
                self.season_team_adjs_received += other_qb_adj
        # return the current value
        return self.current_value

    def update_value(self, value: float, proj: float, gameday: str, season: int, team: str) -> None:
        """Update the quarterback state after a completed game."""
        # update the last game date and season
        self.last_game_date = gameday
        self.last_game_season = season
        self.last_game_team = team
        self.starts += 1
        self.season_starts += 1
        # process the value and error with a progressive discount
        value_adj = prog_disc(
            obs=value, proj=proj, scale=15, alpha=self.params["player_prog_disc_alpha"]
        )
        # update the current value
        # store the current value before it is updated
        old_value = self.current_value
        # the current value does not pass through an s-curve as recent performance in season
        # is more predictive of future performance in the season regardless of a QB's history
        # and our certainty in their overall value
        self.current_value = (
            self.params["player_sf"] * value_adj
            + (1 - self.params["player_sf"]) * self.current_value
        )
        # update the rolling value
        # first, set the rolling sf, which is progressively discounted as a QB has more starts
        # this indicates that their career average, which is regressed to, should be more known
        # and therefore stable
        rolling_sf = self.params["player_career_sf_base"] + s_curve(
            self.params["player_career_sf_height"],
            self.params["player_career_sf_mp"],
            self.starts,
            "down",
        )
        # update the rolling value
        self.rolling_value = rolling_sf * value_adj + (1 - rolling_sf) * self.rolling_value
        # update the variance
        # 𝜎2𝑛=(1−𝛼)𝜎2𝑛−1+𝛼(𝑥𝑛−𝜇𝑛−1)(𝑥𝑛−𝜇𝑛)
        # https://stats.stackexchange.com/questions/6874/exponential-weighted-moving-skewness-kurtosis
        self.current_variance = (
            self.params["player_sf"] * (value - old_value) * (value - self.current_value)
            + (1 - self.params["player_sf"]) * self.current_variance
        )
        # update the qb's adjs received
        self.season_player_adjs_received += self.current_value - old_value

    def regress_value(
        self,
        prev_season_league_avg: float,
    ) -> None:
        """Regress the quarterback value toward career and league baselines."""
        # remove team based adjustments received during the season
        self.current_value = self.current_value - self.season_team_adjs_received
        # calculate the amount to regress to the league average
        # as a QB accumulates more career starts, they or progressively regressed less
        league_regression = s_curve(
            self.params["player_regression_league_height"],
            self.params["player_regression_league_mp"],
            self.starts,
            "down",
        )
        # If the qb did not play significant games during the season (ie was injured or a backup),
        # we do not want to "double regress" them, as they would have also been regressed in the
        # previous season, but did not change significantly during the season.
        # A bad QB could see their value increase despite not playing, while a good QB would see
        # their value decrease.
        # The amount of regression is therefore passed through a second s-curve using season starts
        # as the midpoint variable.
        # calculate value
        league_regression = league_regression * s_curve(
            1,  # height -> 100% represents a full use of the league regression value
            4,  # midpoint -> At 4 starts, only 50% of the league regression value is used
            self.season_starts,  # x -> the number of starts the QB had during the season
            "up",
        )
        # calculate the career regression
        # As a QB accumulates more career starts, they are progressively regressed more to their own
        # career average, which is considered more predictive of their future performance.
        career_regression = s_curve(
            self.params["player_regression_career_height"],
            self.params["player_regression_career_mp"],
            self.starts,
            "up",
        )
        # normalize the combined career and league regression to not exceed 100%
        total_regression = league_regression + career_regression
        if total_regression > 1:
            league_regression = league_regression / total_regression
            career_regression = career_regression / total_regression
        # calculate the new value
        self.current_value = (
            (1 - league_regression - career_regression) * self.current_value
            + (league_regression * prev_season_league_avg)
            + (career_regression * self.rolling_value)
        )
        # reset season based states
        self.season_starts = 0
        self.season_team_adjs_alotted = 0
        self.season_team_adjs_received = 0
        self.season_player_adjs_received = 0
