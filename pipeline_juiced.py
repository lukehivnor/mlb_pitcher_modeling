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
import matplotlib.pyplot as plt

# —— SETUP LOGGING ——
import datetime
now = datetime.datetime.now()

yy = now.year % 100
dating = f"_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}"
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
def load_cache():
    log(f"Loading Statcast cache from {CACHE_FILE}...")
    df = pd.read_csv(CACHE_FILE, parse_dates=['game_date'])
    log(f"Cache loaded: {len(df)} rows")
    return df

# Global cache
df_cache = load_cache()

# Regression targets
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter search spaces
SEQ_LEN_LIST     = [5, 10, 15, 20]
BATCH_SIZE_LIST  = [32, 64, 128, 256]
LR_LIST          = [1e-2, 1e-3, 1e-4]
HIDDEN_LIST      = [32, 64, 128, 256]
LAYER_LIST       = [1, 2, 3]
DROPOUT_RATE     = 0.3  # between-layer dropout
RECURRENT_DROPOUT= 0.3  # unused (PyTorch LSTM doesn't support recurrent_dropout argument)
EPOCHS           = 20
PATIENCE         = 3

class PitcherDataset(Dataset):
    """Creates sequences of length seq_len for LSTM/attention."""
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seq_len = seq_len
        self.X, self.y_event, self.y_reg = [], [], []
        for i in range(len(df) - self.seq_len):
            seq = df[FEATURES].iloc[i:i+self.seq_len].values
            target_idx = i + self.seq_len
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

class MultiTaskAttentionLSTM(nn.Module):
    """LSTM + Self-Attention for multi-task output"""
    def __init__(self, input_dim, hidden_dim, num_layers, num_event_classes, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4,
                                          dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_event = nn.Linear(hidden_dim, num_event_classes)
        self.fc_reg = nn.Linear(hidden_dim, len(REG_TARGETS))

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_out, _ = self.attn(out, out, out)
        h = attn_out[:, -1, :]
        h = self.dropout(h)
        return self.fc_event(h), self.fc_reg(h)


def train_and_evaluate(seq_len, batch_size, lr, hidden, nl):
    log(f"=== Config: SEQ_LEN={seq_len}, BATCH={batch_size}, LR={lr}, HIDDEN={hidden}, LAYERS={nl} ===")

    df = df_cache.copy()
    for tgt in REG_TARGETS:
        df[tgt] = df[tgt].fillna(0)
    df_clean, le_event, encoders, feat_scaler, tgt_scaler = clean_and_scale_features(df)
    if len(df_clean) <= seq_len * 2:
        return None, float('inf')
    split = int(len(df_clean) * 0.8)
    df_train, df_val = df_clean.iloc[:split], df_clean.iloc[split:]

    train_ds = PitcherDataset(df_train, seq_len)
    val_ds = PitcherDataset(df_val, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    y_train = df_train[TARGET_EVENT].values.astype(int)
    n_classes = len(le_event.classes_)
    counts = np.bincount(y_train, minlength=n_classes)
    weights = y_train.shape[0] / counts.astype(float)
    weights[counts == 0] = 0.0
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    crit_event = nn.CrossEntropyLoss(weight=weights_tensor)
    crit_reg = nn.MSELoss()
    model = MultiTaskAttentionLSTM(len(FEATURES), hidden, nl,
                                   len(le_event.classes_), DROPOUT_RATE).to(DEVICE)
    optimz = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimz, mode='min',
                                                     factor=0.5, patience=2)

    best_local = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        for Xb, ye, yr in train_loader:
            Xb, ye, yr = Xb.to(DEVICE), ye.to(DEVICE), yr.to(DEVICE)
            optimz.zero_grad()
            logits, regs = model(Xb)
            loss = crit_event(logits, ye) + crit_reg(regs, yr)
            loss.backward()
            optimz.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for Xv, yev, yrv in val_loader:
                Xv, yev, yrv = Xv.to(DEVICE), yev.to(DEVICE), yrv.to(DEVICE)
                lo, re = model(Xv)
                l_ev = crit_event(lo, yev)
                l_rg = crit_reg(re, yrv)
                losses.append((l_ev + l_rg).item())
        cur_val = np.mean(losses)
        log(f"  Epoch {epoch+1}/{EPOCHS} - val_loss: {cur_val:.4f}")

        scheduler.step(cur_val)
        if cur_val < best_local:
            best_local = cur_val
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log("  Early stopping")
                break

    return best_model, best_local


def hyperparameter_search():
    results = []
    for seq_len in SEQ_LEN_LIST:
        best_loss_for_len = float('inf')
        for batch_size in BATCH_SIZE_LIST:
            for lr in LR_LIST:
                for hidden in HIDDEN_LIST:
                    for nl in LAYER_LIST:
                        model, loss_val = train_and_evaluate(seq_len, batch_size, lr, hidden, nl)
                        results.append((seq_len, batch_size, lr, hidden, nl, loss_val))
                        if loss_val < best_loss_for_len:
                            best_loss_for_len = loss_val
        log(f"Best val loss for SEQ_LEN={seq_len}: {best_loss_for_len:.4f}")
    plt.figure()
    seq_lens = sorted(set(r[0] for r in results))
    best_losses = [min(r[5] for r in results if r[0]==s) for s in seq_lens]
    plt.plot(seq_lens, best_losses, marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Best Validation Loss')
    plt.title('Validation Loss vs. Sequence Length')
    plt.grid(True)
    plot_file = 'val_loss_vs_seq_len.png'
    plt.savefig(plot_file)
    log(f"Saved plot: {plot_file}")
    return results

if __name__ == '__main__':
    log("Starting per-pitcher hyperparameter search...")
    df_pitchers = get_pitchers_to_train(days=2)
    full_cache = df_cache.copy()
    for _, row in df_pitchers.iterrows():
        pitcher_id = int(row.get('mlbam_id', row.get('pitcher', -1)))
        pitcher_name = row.get('pitcher_name', pitcher_id)
        log(f"---- Pitcher: {pitcher_name} (ID={pitcher_id}) ----")
        df_cache = full_cache[full_cache['pitcher'] == pitcher_id]
        if df_cache.empty:
            log(f"No data for pitcher {pitcher_name}, skipping.")
            continue
        results = hyperparameter_search()
        df_res = pd.DataFrame(results,
                              columns=['seq_len','batch_size','lr','hidden','layers','val_loss'])
        csv_file = f"hparams_results_{pitcher_name}{dating}.csv"
        df_res.to_csv(csv_file, index=False)
        log(f"Saved results for {pitcher_name}: {csv_file}")
    df_cache = full_cache
