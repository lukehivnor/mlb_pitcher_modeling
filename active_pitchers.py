from pybaseball import pitching_stats
from datetime import datetime
import pandas as pd


def get_active_pitchers():
    # Define time range
    current_year = datetime.now().year
    years = [current_year - i for i in range(3)]  # Last 3 seasons

    # Accumulate qualified pitchers
    pitcher_counts = {}

    for year in years:
        df = pitching_stats(year, qual=0)  # include all pitchers
        df = df[df['IP'] > 10.0]
        for pid in df['IDfg']:
            pitcher_counts[pid] = pitcher_counts.get(pid, 0) + 1

    # Only include pitchers who show up in all 3 years
    active_pitchers = [pid for pid, count in pitcher_counts.items() if count == 3]

    #for i in active_pitchers:
        #print(i)
    return active_pitchers
