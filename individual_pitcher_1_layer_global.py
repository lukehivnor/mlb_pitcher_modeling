#!/usr/bin/env python3
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# —— SETUP LOGGING ——
now = datetime.datetime.now()
yy = now.year % 100
dating = f"_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}"
log_filename = f"individual_pitcher_1_layer_global{dating}.txt"
log_file = open(log_filename, "w")

def log(msg):
    msg = str(msg)
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# Project imports
from investigate_drops5 import CACHE_FILE
from pitchers_to_train import get_pitchers_to_train
from config import FEATURES, TARGET_EVENT
from clean_and_scale_features import clean_and_scale_features

# Regression targets
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# —— Dataset ——
class GlobalPitcherDataset(Dataset):
    """Yields (seq, pid_idx, event, reg) tuples."""
    def __init__(self, df_by_pitcher, seq_len, pid_to_idx):
        self.X, self.pitcher_idxs, self.y_event, self.y_reg = [], [], [], []
        for pid, df in df_by_pitcher.items():
            idx = pid_to_idx[pid]
            for i in range(len(df) - seq_len):
                seq = df[FEATURES].iloc[i:i+seq_len].values
                tgt = i + seq_len
                self.X.append(seq)
                self.pitcher_idxs.append(idx)
                self.y_event.append(df[TARGET_EVENT].iloc[tgt])
                self.y_reg.append(df[REG_TARGETS].iloc[tgt].values)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=DEVICE)
        self.pitcher_idxs = torch.tensor(self.pitcher_idxs, dtype=torch.long, device=DEVICE)
        self.y_event = torch.tensor(self.y_event, dtype=torch.long, device=DEVICE)
        self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32, device=DEVICE)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.pitcher_idxs[idx], self.y_event[idx], self.y_reg[idx]

# —— Model ——
class GlobalMultiTaskAttentionLSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, num_pitchers,
                 hidden_dim, num_layers, num_event_classes,
                 dropout_rate, n_heads, attn_dropout,
                 embed_dropout, ffn_exp, norm_type):
        super().__init__()
        self.pitcher_emb = nn.Embedding(num_pitchers, emb_dim)
        self.embed_drop = nn.Dropout(embed_dropout)
        self.lstm = nn.LSTM(input_dim + emb_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=n_heads,
                                          dropout=attn_dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim) if norm_type == 'layer' else nn.Identity()
        self.ffn = (nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * ffn_exp),
                        nn.ReLU(),
                        nn.Linear(hidden_dim * ffn_exp, hidden_dim),
                        nn.Dropout(dropout_rate)
                    ) if ffn_exp > 1 else nn.Identity())
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_event = nn.Linear(hidden_dim, num_event_classes)
        self.fc_reg = nn.Linear(hidden_dim, len(REG_TARGETS))

    def forward(self, x, pid_idx):
        emb = self.embed_drop(self.pitcher_emb(pid_idx))
        emb_seq = emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_aug = torch.cat([x, emb_seq], dim=-1)
        out, _ = self.lstm(x_aug)
        attn_out, _ = self.attn(out, out, out)
        h = attn_out[:, -1, :]
        h = self.norm(h)
        h = self.ffn(h)
        h = self.dropout(h)
        return self.fc_event(h), self.fc_reg(h)

# —— Freeze Helper ——
def freeze_global_model(model):
    for name, param in model.named_parameters():
        if not (name.startswith('pitcher_emb')
                or name.startswith('fc_event')
                or name.startswith('fc_reg')):
            param.requires_grad = False
    log("Global model frozen (only embedding + heads trainable)")
    return model

# Two fixed configurations
models_cfg = {
    "unscaled_1_layer_global": {
        "seq_len": 20, "batch_size": 128, "lr": 0.00011259939619198685,
        "hidden": 256, "nl": 1, "emb_dim": 32, "n_heads": 1,
        "attn_do": 0.2, "emb_do": 0.1, "ffn_exp": 4, "norm_type": "layer"
    },
    "transform_1_layer_global": {
        "seq_len": 5, "batch_size": 32, "lr": 0.0009221036158231293,
        "hidden": 64, "nl": 1, "emb_dim": 16, "n_heads": 1,
        "attn_do": 0.2, "emb_do": 0.0, "ffn_exp": 4, "norm_type": "layer"
    }
}

# Training settings
EPOCHS = 20
PATIENCE = 5

# Sinusoidal LR schedule
def lr_lambda(epoch):
    return 0.5 * (1 + np.cos(np.pi * epoch / (EPOCHS - 1)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days back for pitcher list")
    args = parser.parse_args()

    # Load Statcast cache
    log(f"Loading Statcast cache from {CACHE_FILE}...")
    df_cache = pd.read_csv(CACHE_FILE, parse_dates=["game_date"])
    df_cache = df_cache.sort_values("game_date")
    log(f"Cache loaded: {len(df_cache)} rows")

    # Get pitcher IDs
    pids = get_pitchers_to_train(days=args.days)["mlbam_id"].astype(int).tolist()

    # Preprocess and scale
    df_raw = df_cache[df_cache["pitcher"].isin(pids)].copy()
    for tgt in REG_TARGETS:
        df_raw[tgt] = df_raw[tgt].fillna(0)
    df_clean, le, encs, scaler_x, scaler_y = clean_and_scale_features(df_raw)
    df_clean["pitcher"] = df_raw["pitcher"].loc[df_clean.index].values

    # Organize by pitcher
    df_by_pid = {pid: df_clean[df_clean["pitcher"] == pid] for pid in pids}

    # Train each fixed model
    for model_name, cfg in models_cfg.items():
        log(f"=== Training {model_name} ===")

        # Unpack hyperparams
        seq_len   = cfg["seq_len"]
        batch_size= cfg["batch_size"]
        lr        = cfg["lr"]
        hidden    = cfg["hidden"]
        nl        = cfg["nl"]
        emb_dim   = cfg["emb_dim"]
        n_heads   = cfg["n_heads"]
        attn_do   = cfg["attn_do"]
        emb_do    = cfg["emb_do"]
        ffn_exp   = cfg["ffn_exp"]
        norm_type = cfg["norm_type"]

        # Filter pitchers with enough data
        df_filt = {pid: df for pid, df in df_by_pid.items() if len(df) > seq_len * 2}
        pid_to_idx = {pid: i for i, pid in enumerate(df_filt)}

        # Dataset + loader
        dataset = GlobalPitcherDataset(df_filt, seq_len, pid_to_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Build model
        model = GlobalMultiTaskAttentionLSTM(
            input_dim=len(FEATURES), emb_dim=emb_dim,
            num_pitchers=len(pid_to_idx), hidden_dim=hidden,
            num_layers=nl, num_event_classes=len(le.classes_),
            dropout_rate=0.3, n_heads=n_heads,
            attn_dropout=attn_do, embed_dropout=emb_do,
            ffn_exp=ffn_exp, norm_type=norm_type
        ).to(DEVICE)

        # Class-imbalance weights
        y_all = dataset.y_event.cpu().numpy()
        cnts  = np.bincount(y_all, minlength=len(le.classes_))
        wts   = len(y_all) / (cnts.astype(float) + 1e-8)
        wts[cnts == 0] = 0.0
        w_tensor = torch.tensor(wts, dtype=torch.float32, device=DEVICE)

        crit_e = nn.CrossEntropyLoss(weight=w_tensor)
        crit_r = nn.MSELoss()

        # Optimizer & scheduler
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop with early stopping
        best_loss  = float("inf")
        bad_epochs = 0
        best_state = None

        for epoch in range(EPOCHS):
            model.train()
            for Xb, pidb, ye, yr in loader:
                Xb, pidb, ye, yr = Xb.to(DEVICE), pidb.to(DEVICE), ye.to(DEVICE), yr.to(DEVICE)
                optimizer.zero_grad()
                logits, regs = model(Xb, pidb)
                loss = crit_e(logits, ye) + crit_r(regs, yr)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation (real‐unit MSE)
            model.eval()
            losses = []
            with torch.no_grad():
                for Xb, pidb, ye, yr in loader:
                    Xb, pidb, ye, yr = Xb.to(DEVICE), pidb.to(DEVICE), ye.to(DEVICE), yr.to(DEVICE)
                    logits, regs = model(Xb, pidb)
                    closs = crit_e(logits, ye).item()
                    re_np = regs.cpu().numpy()
                    yr_np = yr.cpu().numpy()
                    orig_re = scaler_y.inverse_transform(re_np)
                    orig_yr = scaler_y.inverse_transform(yr_np)
                    rloss = np.mean((orig_re - orig_yr) ** 2)
                    losses.append(closs + rloss)
            val_loss = np.mean(losses)

            log(f"{model_name} Epoch {epoch+1}/{EPOCHS} - val_loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict().copy()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    log("  Early stopping reached")
                    break

        # Load best and save
        model.load_state_dict(best_state)
        full_path = f"models/{model_name}_full{dating}.pt"
        torch.save(model.state_dict(), full_path)
        log(f"Saved full model to {full_path}")

        frozen = freeze_global_model(model)
        frozen_path = f"models/{model_name}_frozen{dating}.pt"
        torch.save(frozen.state_dict(), frozen_path)
        log(f"Saved frozen model to {frozen_path}")

    log("All models trained.")

# —— Main Script ——
if __name__ == "__main__":
    main()
