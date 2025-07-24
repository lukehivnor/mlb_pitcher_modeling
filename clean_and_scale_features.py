import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import cleaning constants
from investigate_drops5 import DROP_ALWAYS, BASE_STATE_COLS, HIT_COLS, RELEASE_COLS, CATEGORICAL_COLS, DESCRIPTION_COL
# Import pipeline constants
from config import FEATURES, TARGET_EVENT

# Define your regression targets here (or import from config if you prefer)
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']

def clean_and_scale_features(df: pd.DataFrame):
    """
    Clean the DataFrame using the same logic as investigate_drops5.py, then
    label-encode and scale both FEATURES and REG_TARGETS.
    Returns:
        df_clean (pd.DataFrame): Cleaned and transformed DataFrame.
        le_event (LabelEncoder): Fitted on TARGET_EVENT.
        encoders (dict): LabelEncoders for all categorical features.
        feat_scaler (StandardScaler): Fitted on FEATURES.
        target_scaler (StandardScaler): Fitted on REG_TARGETS.
    """
    # 1) Copy data
    df_clean = df.copy()

    # 2) Drop always-empty columns
    drop_cols = [c for c in DROP_ALWAYS if c in df_clean.columns]
    df_clean.drop(columns=drop_cols, inplace=True)

    # 3) Drop rows missing any release metrics
    df_clean.dropna(subset=RELEASE_COLS, inplace=True)

    # 4) Fill base-state flags with 0
    df_clean[BASE_STATE_COLS] = df_clean[BASE_STATE_COLS].fillna(0)

    # 5) Fill hit metadata
    df_clean['events'] = df_clean['events'].fillna(df_clean[DESCRIPTION_COL])
    df_clean['hit_location'] = df_clean['hit_location'].fillna(df_clean[DESCRIPTION_COL])
    df_clean['bb_type'] = df_clean['bb_type'].fillna(df_clean[DESCRIPTION_COL])
    df_clean['hit_distance'] = df_clean['hit_distance'].fillna(0)

    # 6) Fill any remaining categorical NAs with 'UNK'
    for col in CATEGORICAL_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('UNK')

    # 7) Drop any rows still containing NA
    df_clean.dropna(inplace=True)

    # 8) Label-encode the event target
    le_event = LabelEncoder()
    df_clean[TARGET_EVENT] = le_event.fit_transform(df_clean[TARGET_EVENT])

    # 9) Label-encode categorical features
    cat_feats = [
        'game_type', 'stand', 'p_throws',
        'hit_location', 'bb_type', 'inning_topbot',
        'description', 'pitch_type', 'pitch_name'
    ]
    encoders = {}
    for feat in cat_feats:
        if feat in df_clean.columns:
            le = LabelEncoder()
            df_clean[feat] = le.fit_transform(df_clean[feat].astype(str))
            encoders[feat] = le

    # 10) Scale numeric features
    feat_scaler = StandardScaler()
    df_clean[FEATURES] = feat_scaler.fit_transform(df_clean[FEATURES])

    # 11) Scale regression targets
    target_scaler = StandardScaler()
    df_clean[REG_TARGETS] = target_scaler.fit_transform(df_clean[REG_TARGETS])

    return df_clean, le_event, encoders, feat_scaler, target_scaler
