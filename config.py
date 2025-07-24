# config.py

# Features used as input to the LSTM
FEATURES = [
    'release_spin_rate', 'release_extension', 'spin_axis',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot',
    'release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z',
    'game_type', 'stand', 'p_throws', 'hit_location', 'bb_type', 'description' , 'pitch_type', 'pitch_name', 'pitch_number'
]
# 'game_type', 'stand', 'p_throws', 'hit_location', 'bb_type', 'inning_topbot', 'description' , 'pitch_type', 'pitch_name'
# The column we’re predicting as a classification target
TARGET_EVENT = 'events'
