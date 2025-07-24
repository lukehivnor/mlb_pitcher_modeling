#!/usr/bin/env python3
import os
import argparse
import datetime
import optuna

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# —— SETUP LOGGING ——
now = datetime.datetime.now()
yy = now.year % 100
dating = f"_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}"
log_filename = f"global_pitcher_search{dating}.txt"
log_file = open(log_filename, "w")

def log(msg):
    msg_str = str(msg)
    print(msg_str)
    log_file.write(msg_str + "\n")
    log_file.flush()

# Project imports
from investigate_drops5 import CACHE_FILE
from pitchers_to_train import get_pitchers_to_train
from config import FEATURES, TARGET_EVENT
from clean_and_scale_features import clean_and_scale_features

# Regression targets
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']
# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameter grids
SEQ_LEN_LIST    = [5, 10, 15, 20]
BATCH_SIZE_LIST = [32, 64, 128, 256]
LR_LIST         = [1e-2, 1e-3, 1e-4]
HIDDEN_LIST     = [32, 64, 128, 256]
LAYER_LIST      = [1, 2, 3]
DROPOUT_LIST    = [0.3]
EMB_DIM_LIST    = [8, 16, 32]
HEADS_LIST      = [1, 2, 4]
ATTN_DO_LIST    = [0.0, 0.1, 0.2]
EMBED_DO_LIST   = [0.0, 0.1]
FFN_EXP_LIST    = [2, 4]
NORM_LIST       = ['layer', 'none']
EPOCHS_GLOBAL   = 20
PATIENCE_GLOBAL = 3

# Ensure models dir
os.makedirs('models', exist_ok=True)

# —— Data Loading ——
def load_cache():
    log(f"Loading Statcast cache from {CACHE_FILE}...")
    df = pd.read_csv(CACHE_FILE, parse_dates=['game_date'])
    log(f"Cache loaded: {len(df)} rows")
    return df

# —— Dataset ——
class GlobalPitcherDataset(Dataset):
    """Dataset yielding (seq, pid_idx, event, reg)"""
    def __init__(self, df_by_pitcher, seq_len, pid_to_idx):
        self.seq_len = seq_len
        self.pid_to_idx = pid_to_idx
        self.X, self.pitcher_idxs, self.y_event, self.y_reg = [], [], [], []
        for pid, df in df_by_pitcher.items():
            idx = pid_to_idx[pid]
            for i in range(len(df) - seq_len):
                seq = df[FEATURES].iloc[i:i+seq_len].values
                tgt_idx = i + seq_len
                self.X.append(seq)
                self.pitcher_idxs.append(idx)
                self.y_event.append(df[TARGET_EVENT].iloc[tgt_idx])
                self.y_reg.append(df[REG_TARGETS].iloc[tgt_idx].values)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=device)
        self.pitcher_idxs = torch.tensor(self.pitcher_idxs, dtype=torch.long, device=device)
        self.y_event = torch.tensor(self.y_event, dtype=torch.long, device=device)
        self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32, device=device)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.pitcher_idxs[idx], self.y_event[idx], self.y_reg[idx]

# —— Model ——
class GlobalMultiTaskAttentionLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, num_pitchers,
                 hidden_dim, num_layers, num_event_classes,
                 dropout_rate, n_heads, attn_dropout,
                 embed_dropout, ffn_exp, norm_type):
        super().__init__()
        self.pitcher_emb = nn.Embedding(num_pitchers, emb_dim)
        self.embed_do = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_dim + emb_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers>1 else 0)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim) if norm_type=='layer' else nn.Identity()
        self.ffn = (nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim*ffn_exp),
                        nn.ReLU(),
                        nn.Linear(hidden_dim*ffn_exp, hidden_dim),
                        nn.Dropout(dropout_rate))
                    if ffn_exp>1 else nn.Identity())
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_event = nn.Linear(hidden_dim, num_event_classes)
        self.fc_reg = nn.Linear(hidden_dim, len(REG_TARGETS))

    def forward(self, x, pid_idx):
        emb = self.embed_do(self.pitcher_emb(pid_idx))
        emb_seq = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_aug = torch.cat([x, emb_seq], dim=-1)
        out, _ = self.lstm(x_aug)
        attn_out, _ = self.attn(out, out, out)
        h = attn_out[:, -1, :]
        h = self.norm(h)
        h = self.ffn(h)
        h = self.dropout(h)
        return self.fc_event(h), self.fc_reg(h)

# —— Freeze ——
def freeze_global_model(model):
    for name, param in model.named_parameters():
        if not (name.startswith('pitcher_emb') or name.startswith('fc_event') or name.startswith('fc_reg')):
            param.requires_grad = False
    log("Global model frozen (only embedding and heads trainable).")
    return model

# —— Hyperparameter Search with Optuna ——
def global_hyperparameter_search(df_cache, pid_list, n_trials):
    log("=== Starting global hyperparameter sweep with Optuna ===")
    # Pre-process once
    df_raw = df_cache[df_cache['pitcher'].isin(pid_list)].copy()
    for tgt in REG_TARGETS:
        df_raw[tgt] = df_raw[tgt].fillna(0)
    # get scalers for inverse-transform
    df_clean, le, encs, scaler_x, scaler_y = clean_and_scale_features(df_raw)
    df_clean['pitcher'] = df_raw['pitcher'].loc[df_clean.index].values
    df_by_pid = {pid: df_clean[df_clean['pitcher']==pid] for pid in pid_list}

    def objective(trial):
        seq_len    = trial.suggest_categorical('seq_len', SEQ_LEN_LIST)
        batch_size = trial.suggest_categorical('batch_size', BATCH_SIZE_LIST)
        lr         = trial.suggest_loguniform('lr', min(LR_LIST), max(LR_LIST))
        hidden     = trial.suggest_categorical('hidden', HIDDEN_LIST)
        nl         = trial.suggest_categorical('nl', LAYER_LIST)
        emb_dim    = trial.suggest_categorical('emb_dim', EMB_DIM_LIST)
        n_heads    = trial.suggest_categorical('n_heads', HEADS_LIST)
        attn_do    = trial.suggest_categorical('attn_do', ATTN_DO_LIST)
        emb_do     = trial.suggest_categorical('emb_do', EMBED_DO_LIST)
        ffn_exp    = trial.suggest_categorical('ffn_exp', FFN_EXP_LIST)
        norm_type  = trial.suggest_categorical('norm_type', NORM_LIST)

        df_filtered = {pid:df for pid,df in df_by_pid.items() if len(df)>seq_len*2}
        if not df_filtered: return float('inf')
        pid_to_idx = {pid:i for i,pid in enumerate(df_filtered)}
        dataset = GlobalPitcherDataset(df_filtered, seq_len, pid_to_idx)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # init model
        model = GlobalMultiTaskAttentionLSTM(
            input_dim=len(FEATURES), emb_dim=emb_dim,
            num_pitchers=len(pid_to_idx), hidden_dim=hidden,
            num_layers=nl, num_event_classes=len(le.classes_),
            dropout_rate=DROPOUT_LIST[0], n_heads=n_heads,
            attn_dropout=attn_do, embed_dropout=emb_do,
            ffn_exp=ffn_exp, norm_type=norm_type
        ).to(device)

        # class-imbalance weights
        y_all = dataset.y_event.cpu().numpy()
        n_cls = len(le.classes_)
        cnts = np.bincount(y_all, minlength=n_cls)
        w    = len(y_all)/(cnts.astype(float)+1e-8)
        w[cnts==0] = 0.0
        w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        crit_e = nn.CrossEntropyLoss(weight=w_tensor)
        crit_r = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # one epoch train
        model.train()
        for Xb, pidb, ye, yr in loader:
            Xb, pidb, ye, yr = Xb.to(device), pidb.to(device), ye.to(device), yr.to(device)
            optimizer.zero_grad()
            lo, re = model(Xb, pidb)
            loss = crit_e(lo, ye) + crit_r(re, yr)
            loss.backward()
            optimizer.step()

        # validation on real-units
        model.eval()
        losses = []
        with torch.no_grad():
            for Xb, pidb, ye, yr in loader:
                Xb, pidb, ye, yr = Xb.to(device), pidb.to(device), ye.to(device), yr.to(device)
                lo, re = model(Xb, pidb)
                closs = crit_e(lo, ye).item()
                re_np = re.cpu().numpy()
                yr_np = yr.cpu().numpy()
                orig_re = scaler_y.inverse_transform(re_np)
                orig_yr = scaler_y.inverse_transform(yr_np)
                rloss = np.mean((orig_re-orig_yr)**2)
                losses.append(closs + rloss)
        val_loss = np.mean(losses)
        trial.report(val_loss, 0)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    log(f"Optuna best score {study.best_value:.4f}, params: {best_params}")
    best_cfg = (
        best_params['seq_len'], best_params['batch_size'], best_params['lr'],
        best_params['hidden'], best_params['nl'], best_params['emb_dim'],
        best_params['n_heads'], best_params['attn_do'], best_params['emb_do'],
        best_params['ffn_exp'], best_params['norm_type']
    )
    return best_cfg

# —— Final Training ——
def train_final_global(df_cache, pid_list, best_cfg):
    log("=== Training final global model ===")
    seq_len, batch_size, lr, hidden, nl, emb_dim, n_heads, attn_do, emb_do, ffn_exp, norm_type = best_cfg
    df_raw = df_cache[df_cache['pitcher'].isin(pid_list)].copy()
    for tgt in REG_TARGETS:
        df_raw[tgt] = df_raw[tgt].fillna(0)
    df_clean, le, encs, scaler_x, scaler_y = clean_and_scale_features(df_raw)
    df_clean['pitcher'] = df_raw['pitcher'].loc[df_clean.index].values
    df_by_pid = {pid:df_clean[df_clean['pitcher']==pid] for pid in pid_list}
    df_filtered = {pid:df for pid,df in df_by_pid.items() if len(df)>seq_len*2}
    pid_to_idx = {pid:i for i,pid in enumerate(df_filtered)}
    dataset = GlobalPitcherDataset(df_filtered, seq_len, pid_to_idx)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GlobalMultiTaskAttentionLSTM(
        input_dim=len(FEATURES), emb_dim=emb_dim,
        num_pitchers=len(pid_to_idx), hidden_dim=hidden,
        num_layers=nl, num_event_classes=len(le.classes_),
        dropout_rate=DROPOUT_LIST[0], n_heads=n_heads,
        attn_dropout=attn_do, embed_dropout=emb_do,
        ffn_exp=ffn_exp, norm_type=norm_type
    ).to(device)

    # class‐imbalance weights
    y_train = df_filtered[next(iter(pid_to_idx))][TARGET_EVENT].values.astype(int)
    n_cls   = len(le.classes_)
    cnts    = np.bincount(y_train, minlength=n_cls)
    wgt     = len(y_train)/(cnts.astype(float)+1e-8)
    wgt[cnts==0] = 0.0
    w_tensor= torch.tensor(wgt, dtype=torch.float32, device=device)
    crit_e = nn.CrossEntropyLoss(weight=w_tensor)
    crit_r = nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=lr)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=PATIENCE_GLOBAL)

    for epoch in range(EPOCHS_GLOBAL):
        model.train()
        for Xb, pidb, ye, yr in loader:
            Xb, pidb, ye, yr = Xb.to(device), pidb.to(device), ye.to(device), yr.to(device)
            optimizer.zero_grad()
            lo, re = model(Xb, pidb)
            loss = crit_e(lo, ye) + crit_r(re, yr)
            loss.backward()
            optimizer.step()
        model.eval()
        losses=[]
        with torch.no_grad():
            for Xb, pidb, ye, yr in loader:
                Xb, pidb, ye, yr = Xb.to(device), pidb.to(device), ye.to(device), yr.to(device)
                lo, re = model(Xb, pidb)
                closs  = crit_e(lo, ye).item()
                re_np  = re.cpu().numpy()
                yr_np  = yr.cpu().numpy()
                orig_re= scaler_y.inverse_transform(re_np)
                orig_yr= scaler_y.inverse_transform(yr_np)
                rloss  = np.mean((orig_re-orig_yr)**2)
                losses.append(closs + rloss)
        val_loss = np.mean(losses)
        log(f"Epoch {epoch+1}/{EPOCHS_GLOBAL} - val_loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    # save full model
    full_path = os.path.join('models', f'global_model_full{dating}.pt')
    torch.save(model.state_dict(), full_path)
    log(f"Saved full global model to {full_path}")
    return model

# —— Main ——
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30, help='Days back for pitcher list')
    parser.add_argument('--trials',type=int,default=50,help='Number of hyperparameter trials')
    args = parser.parse_args()

    df_cache    = load_cache()
    pitcher_ids = get_pitchers_to_train(days=args.days)['mlbam_id'].astype(int).tolist()

    best_cfg = global_hyperparameter_search(df_cache, pitcher_ids, n_trials=args.trials)
    model_full = train_final_global(df_cache, pitcher_ids, best_cfg)
    model_frozen=freeze_global_model(model_full)
    frozen_path=os.path.join('models',f'global_model_frozen{dating}.pt')
    torch.save(model_frozen.state_dict(), frozen_path)
    log(f"Saved frozen global model to {frozen_path}")
    log("Global training pipeline complete.")
