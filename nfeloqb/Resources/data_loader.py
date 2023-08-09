## IMPORTS ##
import pandas as pd
import numpy
import pathlib

class DataLoader():
    ## this class retrieves, loads, and merges data ##
    def __init__(self):
        ## dfs we want to output ##
        self.model_df = None
        self.games_df = None
        ## locations of external data ##
        self.player_stats_url = 'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv.gz'
        self.player_info_url = 'https://github.com/nflverse/nflverse-data/releases/download/players/players.csv'
        self.game_data_url = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
        ## repl dicts ##
        self.player_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
        }
        self.games_file_repl = {
            'LA': 'LAR',
            'LV': 'OAK',
            'STL': 'LAR',
            'SD': 'LAC',
        }
        ## variabbles
        self.stat_cols = [
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]
        ## load file path ##
        data_folder = pathlib.Path(__file__).parent.parent.resolve()
        self.missing_draft_data = '{0}/Manual Data/missing_draft_data.csv'.format(data_folder)
        ## get data on load ##
        self.pull_data()
        
    def retrieve_player_stats(self):
        ## download stats from nflverse ##
        ## these are pre-aggregated from the pbp, which saves time/compute ##
        try:
            df = pd.read_csv(
                self.player_stats_url,
                compression='gzip'
            )
            ## only  qbs are relevant ##
            df=df[
                df['position'] == 'QB'
            ].copy()
            ## file only has recent abbreviation of team name. Standardize so we can join ##
            df['recent_team'] = df['recent_team'].replace(self.player_file_repl)
            df = df.rename(columns={
                'recent_team': 'team',
            })
            return df
        except Exception as e:
            print('     Error retrieving player stats: ' + str(e))
            return None
    
    def retrieve_player_meta(self, df):
        ## get player meta and add it to the stats ##
        ## will be used for draft position and joining ##
        ## the meta file has missing draft data, which has been manually compiled ##
        ## and will be added ##
        def add_missing_draft_data(df):
            ## load missing draft data ##
            missing_draft = pd.read_csv(
                self.missing_draft_data,
                index_col=0
            )
            ## groupby id to ensure no dupes ##
            missing_draft = missing_draft.groupby(['player_id']).head(1)
            ## rename the cols, which will fill if main in NA ##
            missing_draft = missing_draft.rename(columns={
                'rookie_year' : 'rookie_year_fill',
                'draft_number' : 'draft_number_fill',
                'entry_year' : 'entry_year_fill',
                'birth_date' : 'birth_date_fill',
            })
            ## add to data ##
            df = pd.merge(
                df,
                missing_draft[[
                    'player_id', 'rookie_year_fill', 'draft_number_fill',
                    'entry_year_fill', 'birth_date_fill'
                ]],
                on=['player_id'],
                how='left'
            )
            ## fill in missing data ##
            for col in [
                'rookie_year', 'draft_number', 'entry_year', 'birth_date'
            ]:
                ## fill in missing data ##
                df[col] = df[col].combine_first(df[col + '_fill'])
                ## and then drop fill col ##
                df = df.drop(columns=[col + '_fill'])
            ## return ##
            return df
        ## get player meta ##
        try:
            meta = pd.read_csv(
                self.player_info_url
            )
            meta = meta.groupby(['gsis_id']).head(1)
            ## add to df ##
            df = pd.merge(
                df,
                meta[[
                    'gsis_id', 'first_name', 'last_name',
                    'birth_date', 'rookie_year', 'entry_year',
                    'draft_number'
                ]].rename(columns={
                    'gsis_id': 'player_id',
                }),
                on=['player_id'],
                how='left'
            )
            ## add missing draft data ##
            df = add_missing_draft_data(df)
            ## return ##
            return df
        except Exception as e:
            print('     Error retrieving player info: ' + str(e))
            return None
    
    def add_game_data(self, df):
        ## add game data ##
        try:
            game = pd.read_csv(
                self.game_data_url
            )
            ## replace team names ##
            game['home_team'] = game['home_team'].replace(self.games_file_repl)
            game['away_team'] = game['away_team'].replace(self.games_file_repl)
            ## use replaced home and away names to reconstitute the game id ##
            game['game_id'] = (
                game['season'].astype(str) + '_' +
                game['week'].astype(str).str.zfill(2) + '_' +
                game['away_team'] + '_' +
                game['home_team']
            )
            ## games will be used in the future so add to class ##
            self.games = game.copy()
            ## flatten ##
            game_flat = pd.concat([
                game[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name',
                ]].rename(columns={
                    'home_team' : 'team',
                    'home_qb_id' : 'starter_id',
                    'home_qb_name' : 'starter_name',
                    'away_team' : 'opponent',
                    'away_qb_id' : 'opponent_starter_id',
                    'away_qb_name' : 'opponent_starter_name',
                }),
                game[[
                    'game_id', 'gameday', 'season', 'week',
                    'home_team', 'away_team',
                    'home_qb_id', 'home_qb_name',
                    'away_qb_id', 'away_qb_name',
                ]].rename(columns={
                    'away_team' : 'team',
                    'away_qb_id' : 'starter_id',
                    'away_qb_name' : 'starter_name',
                    'home_team' : 'opponent',
                    'home_qb_id' : 'opponent_starter_id',
                    'home_qb_name' : 'opponent_starter_name',
                })
            ])
            ## add to df ##
            df = pd.merge(
                df,
                game_flat,
                on=['season','week','team'],
                how='left'
            )
            ## return ##
            return df
        except Exception as e:
            print('     Error adding game data: ' + str(e))
            return None
    
    ## funcs for calculating value and formatting model file ##
    def aggregate_team_stats(self, df, team_field='team'):
        ## aggregates the individual player file into a team file ##
        ## team field denotes whether to use team or opponent ##
        return df.groupby(['game_id', 'season', 'week', 'gameday', team_field]).agg(
            completions = ('completions', 'sum'),
            attempts = ('attempts', 'sum'),
            passing_yards = ('passing_yards', 'sum'),
            passing_tds = ('passing_tds', 'sum'),
            interceptions = ('interceptions', 'sum'),
            sacks = ('sacks', 'sum'),
            carries = ('carries', 'sum'),
            rushing_yards = ('rushing_yards', 'sum'),
            rushing_tds = ('rushing_tds', 'sum'),
        ).reset_index().rename(columns={
            team_field : 'team'
        })
    
    def iso_top_passer(self, df):
        ## So as not to update the rating of a QB who had few passes, only include
        ## the top passer ##
        return df.sort_values(
            by=['game_id', 'attempts'],
            ascending=[True, False]
        ).groupby(['game_id', 'team']).head(1).reset_index(drop=True)
    
    def format_top_passer(self, df):
        ## add the start number to the top passer and get rid of unecessary fields ##
        ## note, since we arent pre-loading the existing CSV with data before 1999, this number ##
        ## is an approximation ##
        ## since we will eventually throw out data pre-2022, this is fine (probably) ##
        df['start_number'] = df.groupby(['player_id']).cumcount() + 1
        return df[[
            'game_id', 'season', 'week', 'gameday', 'team', 'opponent', 'player_id', 'player_name', 'player_display_name',
            'start_number', 'rookie_year', 'entry_year', 'draft_number',
            'completions', 'attempts', 'passing_yards', 'passing_tds',
            'interceptions', 'sacks', 'carries', 'rushing_yards', 'rushing_tds'
        ]].copy()
    
    def calculate_raw_value(self, df):
        ## takes a df, with properly named fields and returns a series w/ VALUE ##
        ## formula for reference ##
        ## https://fivethirtyeight.com/methodology/how-our-nfl-predictions-work/ ##
        ##      -2.2 * Pass Attempts +
        ##         3.7 * Completions +
        ##       (Passing Yards / 5) +
        ##        11.3 * Passing TDs –
        ##      14.1 * Interceptions –
        ##          8 * Times Sacked –
        ##       1.1 * Rush Attempts +
        ##       0.6 * Rushing Yards +
        ##        15.9 * Rushing TDs
        return (
            -2.2 * df['attempts'] +
            3.7 * df['completions'] +
            (df['passing_yards'] / 5) +
            11.3 * df['passing_tds'] -
            14.1 * df['interceptions'] -
            8 * df['sacks'] -
            1.1 * df['carries'] +
            0.6 * df['rushing_yards'] +
            15.9 * df['rushing_tds']
        )
    
    def pull_data(self):
        ## wrapper for all the above functions ##
        print('Retrieving nflverse data...')
        df = self.retrieve_player_stats()
        while df is not None:
            ## data retrieval ##
            df = self.retrieve_player_meta(df)
            df = self.add_game_data(df)
            ## merge and format ##
            ## team stats ##
            df_team = self.aggregate_team_stats(df)
            df_team['team_VALUE'] = self.calculate_raw_value(df_team)
            df_team = df_team.drop(columns=self.stat_cols)
            ## df ##
            df = self.iso_top_passer(df)
            df = self.format_top_passer(df)
            df['player_VALUE'] = self.calculate_raw_value(df)
            df = df.drop(columns=self.stat_cols)
            ## create model file ##
            df = pd.merge(
                df,
                df_team[['game_id', 'team', 'team_VALUE']],
                on=['game_id', 'team'],
                how='left'
            )
            self.model_df = df.copy()
            print('     Successfully retrived and stored')
            ## end loop ##
            df = None
