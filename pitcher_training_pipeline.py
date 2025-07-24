import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

import datetime
now = datetime.datetime.now()
yy = now.year % 100
log_filename = f"pitcher_trainer_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}.txt"
log_file = open(log_filename, "w")

def log(msg):
    msg_str = str(msg)
    print(msg_str)
    log_file.write(msg_str + "\n")
    log_file.flush()

# Use cached Statcast data instead of live query
from investigate_drops5 import CACHE_FILE
from pitchers_to_train import get_pitchers_to_train
# Shared configuration and cleaning pipeline
from config import FEATURES, TARGET_EVENT
from clean_and_scale_features import clean_and_scale_features

# Load full cached dataset once
log(f"Loading Statcast cache from {CACHE_FILE}...")
df_cache = pd.read_csv(CACHE_FILE, parse_dates=['game_date'])
log(f"Cache loaded: {len(df_cache)} rows")

# Regression targets
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']
SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 3
# Hyperparameter grid
LR_LIST = [1e-2, 1e-4]
HIDDEN_LIST = [64, 128]
LAYER_LIST = [1, 2]
DROPOUT_RATE = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PitcherDataset(Dataset):
    """Creates sequences of length SEQ_LEN for LSTM."""
    def __init__(self, df: pd.DataFrame):
        self.X, self.y_event, self.y_reg = [], [], []
        for i in range(len(df) - SEQ_LEN):
            seq = df[FEATURES].iloc[i:i+SEQ_LEN].values
            target_idx = i + SEQ_LEN
            self.X.append(seq)
            self.y_event.append(df[TARGET_EVENT].iloc[target_idx])
            self.y_reg.append(df[REG_TARGETS].iloc[target_idx].values)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y_event = torch.tensor(self.y_event, dtype=torch.long)
        self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_event[idx], self.y_reg[idx]


class MultiTaskLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_event_classes, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_event = nn.Linear(hidden_dim, num_event_classes)
        self.fc_reg = nn.Linear(hidden_dim, len(REG_TARGETS))

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = self.dropout(h)
        return self.fc_event(h), self.fc_reg(h)


def train_and_predict_for_pitcher(row: pd.Series) -> dict:
    log("--- train_and_predict_for_pitcher START ---")
    pitcher_id = int(row.get('mlbam_id', -1))

    df = df_cache[df_cache['pitcher'] == pitcher_id]
    if df.empty:
        log("No cached data for this pitcher. Returning empty.")
        return {}

    for tgt in REG_TARGETS:
        df[tgt] = df[tgt].fillna(0)

    log("Cleaning, encoding, and scaling features...")
    # Now returns target scaler as well
    df_clean, le_event, encoders, feat_scaler, tgt_scaler = clean_and_scale_features(df)

    if len(df_clean) <= SEQ_LEN * 2:
        log(f"Not enough records ({len(df_clean)}) for SEQ_LEN={SEQ_LEN}. Returning empty.")
        return {}

    # Split into train/val
    split = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split]
    df_val = df_clean.iloc[split:]

    train_ds = PitcherDataset(df_train)
    val_ds = PitcherDataset(df_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Class imbalance weights
        # Class imbalance weights — zero out entirely missing classes
    y_train = df_train[TARGET_EVENT].values.astype(int)
    n_classes = len(le_event.classes_)  # should be 12
    counts = np.bincount(y_train, minlength=n_classes)

    # inverse‐frequency weighting
    weights = y_train.shape[0] / counts.astype(float)

    # zero‐weight any class that never appears
    weights[counts == 0] = 0.0

    weights_tensor = torch.tensor(weights,
                                  dtype=torch.float32,
                                  device=DEVICE)

    crit_event_base = nn.CrossEntropyLoss
    crit_reg        = nn.MSELoss()


    best_model = None
    best_score = float('inf')

    # Hyperparameter sweep
    for lr in LR_LIST:
        for hidden in HIDDEN_LIST:
            for nl in LAYER_LIST:
                log(f"Training with lr={lr}, hidden={hidden}, layers={nl}")
                model = MultiTaskLSTM(len(FEATURES), hidden, nl,
                                      len(le_event.classes_), DROPOUT_RATE).to(DEVICE)
                crit_event = crit_event_base(weight=weights_tensor)
                optimz = optim.Adam(model.parameters(), lr=lr)

                val_loss_history = []
                patience = 0
                best_local = float('inf')
                for epoch in range(EPOCHS):
                    # Train
                    model.train()
                    for Xb, ye, yr in train_loader:
                        Xb, ye, yr = Xb.to(DEVICE), ye.to(DEVICE), yr.to(DEVICE)
                        optimz.zero_grad()
                        logits, regs = model(Xb)
                        loss = crit_event(logits, ye) + crit_reg(regs, yr)
                        loss.backward()
                        optimz.step()

                    # Validate
                    model.eval()
                    val_losses = []
                    val_preds, val_targets = [], []
                    with torch.no_grad():
                        for Xv, yev, yrv in val_loader:
                            Xv, yev, yrv = Xv.to(DEVICE), yev.to(DEVICE), yrv.to(DEVICE)
                            lo, re = model(Xv)
                            le = crit_event(lo, yev)
                            lr_m = crit_reg(re, yrv)
                            val_losses.append((le + lr_m).item())
                            val_preds.append(torch.argmax(lo, dim=1).cpu())
                            val_targets.append(yev.cpu())
                    cur_val = np.mean(val_losses)
                    val_loss_history.append(cur_val)
                    log(f" Val epoch {epoch+1}/{EPOCHS}, loss: {cur_val:.4f}")
                    # Early stopping
                    if cur_val < best_local:
                        best_local = cur_val
                        patience = 0
                        best_epoch_model = copy.deepcopy(model)
                    else:
                        patience += 1
                        if patience >= PATIENCE:
                            log(" Early stopping")
                            break

                # Compare hyperparams
                if best_local < best_score:
                    best_score = best_local
                    best_model = best_epoch_model

    # Use best model for final prediction
    log("Running prediction on last sequence with best model")
    seq = torch.tensor(df_clean[FEATURES].values[-SEQ_LEN:], dtype=torch.float32)
    seq = seq.unsqueeze(0).to(DEVICE)
    best_model.eval()
    with torch.no_grad():
        logits, regs = best_model(seq)
        ev_idx = torch.argmax(logits, dim=1).item()
        event = le_event.inverse_transform([ev_idx])[0]
        reg_scaled = regs.cpu().numpy()[0]
        reg_vals = tgt_scaler.inverse_transform(reg_scaled.reshape(1, -1))[0]

    result = {
        'pitcher_name': row['pitcher_name'],
        'predicted_event': event,
        'predicted_hit_distance': float(reg_vals[0]),
        'predicted_launch_speed': float(reg_vals[1]),
        'predicted_launch_angle': float(reg_vals[2])
    }
    log(f"Result: {result}")
    log("--- train_and_predict_for_pitcher END ---\n")
    return result


def process_pitchers(days: int = 2) -> pd.DataFrame:
    log("=== process_pitchers START ===")
    df_pitchers = get_pitchers_to_train(days)
    results = []
    for _, r in df_pitchers.iterrows():
        out = train_and_predict_for_pitcher(r)
        if out:
            results.append(out)
    df_results = pd.DataFrame(results)
    log(f"=== process_pitchers END: results shape {df_results.shape} ===")
    return df_results

if __name__ == '__main__':
    log(process_pitchers(days=2))