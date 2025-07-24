import pandas as pd
from typing import List, Optional
from datetime import datetime
from pybaseball import playerid_lookup
from active_pitchers import get_active_pitchers
from probable_pitchers import get_upcoming_starters


def get_pitchers_to_train(days: int = 2) -> pd.DataFrame:
    """
    Fetch upcoming probable pitchers, enrich with Fangraphs and MLBAM IDs,
    filter to active pitchers, and return game info.

    :param days: Number of days ahead to fetch (default 2)
    :return: DataFrame with columns:
             ['pitcher_name','fangraphs_id','mlbam_id','side','game_date','game_time']
    """
    # Fetch active pitcher FG IDs
    active_fg_ids = get_active_pitchers()

    # Fetch upcoming starters
    df_games = get_upcoming_starters(days=days)
    if df_games is None or df_games.empty:
        return pd.DataFrame(columns=[
            'pitcher_name','fangraphs_id','mlbam_id','side','game_date','game_time'
        ])

    # Normalize game_time and date
    df_games['game_time'] = pd.to_datetime(df_games['game_time'], errors='coerce')
    df_games['game_date'] = pd.to_datetime(df_games['date'], errors='coerce')

    records: List[dict] = []
    for _, row in df_games.iterrows():
        for side in ['away', 'home']:
            pname: Optional[str] = row.get(f'{side}_pitcher')
            if not pname or pd.isna(pname):
                continue

            # Lookup IDs via pybaseball
            parts = pname.split()
            first, last = parts[0], parts[-1]
            id_df = playerid_lookup(last, first)
            if id_df.empty:
                fg_id = None
                mlbam_id_lookup = None
            else:
                fg_id = id_df['key_fangraphs'].iloc[0]
                mlbam_id_lookup = id_df['key_mlbam'].iloc[0]

            # Only include if Fangraphs ID is in active list
            if fg_id not in active_fg_ids:
                continue

            # Use API MLBAM ID if available, else lookup
            mlbam_api = row.get(f'{side}_id')
            mlbam_id = int(mlbam_api) if pd.notna(mlbam_api) else mlbam_id_lookup  # type: ignore

            records.append({
                'pitcher_name': pname,
                'fangraphs_id': fg_id,
                'mlbam_id': mlbam_id,
                'side': side,
                'game_date': row['game_date'],
                'game_time': row['game_time']
            })

    return pd.DataFrame(records)


if __name__ == '__main__':
    df = get_pitchers_to_train()
    print(df)
