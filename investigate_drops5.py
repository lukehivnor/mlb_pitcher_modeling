# investigate_drops5_clean.py
# Cleans Statcast data per specified fill/drop logic for upcoming pitchers

import os
import datetime
import warnings
import pandas as pd
from pitchers_to_train import get_pitchers_to_train

SEQ_LEN = 10
# —— SILENCE FUTURE WARNINGS ——
warnings.filterwarnings("ignore")
import datetime
# —— SETUP LOG FILE ——
now = datetime.datetime.now()
yy = now.year % 100
log_filename = f"investigate_drops5_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}.txt"
log_file = open(log_filename, "w")

def log(msg: str):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# —— CONFIGURATION ——
CACHE_FILE = 'statcast_3yr.csv'
# all columns
'''pitch_type	game_date	release_speed	release_pos_x	release_pos_z	player_name	batter	pitcher	events	description	spin_dir	spin_rate_deprecated	break_angle_deprecated	break_length_deprecated	zone	des	game_type	stand	p_throws	home_team	away_team	type	hit_location	bb_type	balls	strikes	game_year	pfx_x	pfx_z	plate_x	plate_z	on_3b	on_2b	on_1b	outs_when_up	inning	inning_topbot	hc_x	hc_y	tfs_deprecated	tfs_zulu_deprecated	umpire	sv_id	vx0	vy0	vz0	ax	ay	az	sz_top	sz_bot	hit_distance	launch_speed	launch_angle	effective_speed	release_spin_rate	release_extension	game_pk	fielder_2	fielder_3	fielder_4	fielder_5	fielder_6	fielder_7	fielder_8	fielder_9	release_pos_y	estimated_ba_using_speedangle	estimated_woba_using_speedangle	woba_value	woba_denom	babip_value	iso_value	launch_speed_angle	at_bat_number	pitch_number	pitch_name	home_score	away_score	bat_score	fld_score	post_away_score	post_home_score	post_bat_score	post_fld_score	if_fielding_alignment	of_fielding_alignment	spin_axis	delta_home_win_exp	delta_run_exp	bat_speed	swing_length	estimated_slg_using_speedangle	delta_pitcher_run_exp	hyper_speed	home_score_diff	bat_score_diff	home_win_exp	bat_win_exp	age_pit_legacy	age_bat_legacy	age_pit	age_bat	n_thruorder_pitcher	n_priorpa_thisgame_player_at_bat	pitcher_days_since_prev_game	batter_days_since_prev_game	pitcher_days_until_next_game	 batter_days_until_next_game	api_break_z_with_gravity	api_break_x_arm	api_break_x_batter_in	arm_angle	attack_angle	attack_direction	swing_path_tilt	intercept_ball_minus_batter_pos_x_inches	intercept_ball_minus_batter_pos_y_inches
'''
# Columns to drop always
DROP_ALWAYS = [
    "spin_dir",
    "spin_rate_deprecated",
    "break_angle_deprecated",
    "break_length_deprecated",
    "tfs_deprecated",
    "tfs_zulu_deprecated",
    "pitcher_days_since_prev_game",
    "batter_days_since_prev_game",
    "pitcher_days_until_next_game",
    "batter_days_until_next_game",
]
DROP_ALWAYS += [
    'attack_angle', 'attack_direction', 'babip_value', 'bat_speed',
    'hc_x', 'hc_y', 'hyper_speed',
    'estimated_ba_using_speedangle', 'estimated_slg_using_speedangle', 'estimated_woba_using_speedangle',
    'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches',
    'iso_value', 
    'swing_length', 'swing_path_tilt', 'sv_id', 'umpire',
    'woba_denom', 'woba_value', 'arm_angle', 'if_fielding_alignment', 'of_fielding_alignment', 'effective_speed'
]
# 'launch_angle', 'launch_speed', 'launch_speed_angle',
# Base-state flags
BASE_STATE_COLS = [
    "on_3b",
    "on_2b",
    "on_1b",
    "outs_when_up",
    "inning",
    "inning_topbot",
]

# Hit-specific metadata
HIT_COLS = [
    "hit_location",
    "bb_type",
    "events",
    "hit_distance",
]

# Release and trajectory metrics
RELEASE_COLS = [
    "release_speed",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
]

# Categorical features
CATEGORICAL_COLS = [
    "game_type",
    "stand",
    "p_throws",
]

# Description field used to fill events and hit data
DESCRIPTION_COL = "description"

# Directory for output CSVs (commented out by default)
out_dir = f"pitcher_data_{now.month}_{now.day}_{yy}"
# os.makedirs(out_dir, exist_ok=True)


def clean_for_pitcher(df: pd.DataFrame, pid: int, name: str):
    sub = df[df['pitcher'] == pid].copy()
    total_before = len(sub)
    log(f"\n=== {name} (ID: {pid}) ===")
    log(f"Total records before cleaning: {total_before}")

    # Drop always-empty columns
    drop_cols = [c for c in DROP_ALWAYS if c in sub.columns]
    sub.drop(columns=drop_cols, inplace=True)
    log(f"Dropped columns: {drop_cols}")

    # 1) Drop rows missing any release metrics
    before_release_drop = len(sub)
    sub.dropna(subset=RELEASE_COLS, inplace=True)
    after_release_drop = len(sub)
    log(f"Dropped {before_release_drop - after_release_drop} rows missing RELEASE_COLS ({RELEASE_COLS})")

    # 2) Fill base-state flags with 0
    sub[BASE_STATE_COLS] = sub[BASE_STATE_COLS].fillna(0)
    log(f"Filled NA in BASE_STATE_COLS with 0")

    # 3) Fill hit metadata using description or defaults
    # events: fill with description
    sub['events'] = sub['events'].fillna(sub[DESCRIPTION_COL])
    # hit_location: fill with description
    sub['hit_location'] = sub['hit_location'].fillna(sub[DESCRIPTION_COL])
    # bb_type: fill with description
    sub['bb_type'] = sub['bb_type'].fillna(sub[DESCRIPTION_COL])
    # hit_distance: fill with 0
    sub['hit_distance'] = sub['hit_distance'].fillna(0)
    log(f"Filled HIT_COLS ({HIT_COLS}) using description and defaults")

    # 4) Fill any remaining categorical NAs with 'UNK'
    for col in CATEGORICAL_COLS:
        if col in sub.columns:
            sub[col] = sub[col].fillna('UNK')
    log(f"Filled NA in CATEGORICAL_COLS with 'UNK'")

    # 5a) DEBUG: find all rows that still have any NA, and log their indices + missing cols
    mask = sub.isnull().any(axis=1)
    if mask.any():
        na_rows = sub[mask]
        for idx, row in na_rows.iterrows():
            missing = row.index[row.isnull()].tolist()
            log(f"Dropping row {idx!r}: missing columns {missing}")
    else:
        log("No rows with NA remaining before final drop")

    # 5b) Now perform the actual drop
    before_final_drop = len(sub)
    sub.dropna(inplace=True)
    after_final_drop = len(sub)
    log(f"Dropped {before_final_drop - after_final_drop} rows still containing NA after cleaning")

    # 6) Report final count and skip if below threshold
    log(f"Final valid rows for {name}: {after_final_drop}")
    if after_final_drop < SEQ_LEN:
        log(f"Insufficient records ({after_final_drop}) for SEQ_LEN={SEQ_LEN}, skipping pitcher")
        return

    # 7) Show a sample of cleaned data
    log("Sample of cleaned data (first 5 rows):")
    log(sub.head(5).to_string(index=False))

    # (Optional) write cleaned data to CSV per pitcher
    # fname = os.path.join(out_dir, f"{name.replace(' ', '_')}_{now.month}_{now.day}_{yy}.csv")
    # sub.to_csv(fname, index=False)
    # log(f"Wrote cleaned data to {fname}")


def main():
    # Load cache
    log("Loading full Statcast cache...")
    df = pd.read_csv(CACHE_FILE, parse_dates=["game_date"])
    log(f"Total records loaded: {len(df)}")

    log("Loading upcoming starters...")
    starters = get_pitchers_to_train(days=2)
    log(f"Found {len(starters)} pitchers to clean")

    # Process each starter
    for _, row in starters.iterrows():
        pid = int(row["mlbam_id"])
        name = row["pitcher_name"]
        clean_for_pitcher(df, pid, name)

    log_file.close()


if __name__ == "__main__":
    main()
