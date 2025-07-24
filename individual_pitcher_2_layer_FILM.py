#!/usr/bin/env python3
import os
import glob
import argparse
import datetime
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, accuracy_score

# —— SETUP LOGGING ——
now = datetime.datetime.now()
yy = now.year % 100
dating = f"_{now.month}_{now.day}_{yy}_{now.hour}_{now.minute}"
log_filename = f"individual_pitcher_2_layer_FILM{dating}.txt"
log_file = open(log_filename, "w")
REG_TARGETS = ['hit_distance', 'launch_speed', 'launch_angle']

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
# Load global model definition for weight shapes
from global_pitcher_model_search import GlobalMultiTaskAttentionLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fine-tuning hyperparams
EPOCHS_FINE = 20
BATCH_SIZE_FINE = 32
INITIAL_LR = 1e-2
PATIENCE_FINE = 3

# Ensure models dir
os.makedirs('models', exist_ok=True)

# —— Data Loading ——
def load_cache():
    log(f"Loading Statcast cache from {CACHE_FILE}...")
    df = pd.read_csv(CACHE_FILE, parse_dates=['game_date'])
    df = df.sort_values('game_date')
    log(f"Cache loaded: {len(df)} rows")
    return df

# —— Dataset ——
class PitcherDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.seq_len = seq_len
        self.X, self.y_event, self.y_reg = [], [], []
        for i in range(len(df) - seq_len):
            seq = df[FEATURES].iloc[i:i+seq_len].values
            tgt = i + seq_len
            self.X.append(seq)
            self.y_event.append(df[TARGET_EVENT].iloc[tgt])
            self.y_reg.append(df[REG_TARGETS].iloc[tgt].values)
        self.X = torch.tensor(self.X, dtype=torch.float32, device=device)
        self.y_event = torch.tensor(self.y_event, dtype=torch.long, device=device)
        self.y_reg = torch.tensor(self.y_reg, dtype=torch.float32, device=device)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_event[idx], self.y_reg[idx]

# —— Two-Layer FiLM LSTM Model ——
class TwoLayerFiLMLSTM(nn.Module):
    def __init__(self, base_model, hidden_dim, recurrent_dropout):
        super().__init__()
        # base_model: pre-loaded GlobalMultiTaskAttentionLSTM
        # Reuse embedding and LSTM1, attn, norm, ffn, heads from base_model
        self.pitcher_emb = base_model.pitcher_emb
        self.embed_drop = base_model.embed_do
        self.lstm1 = base_model.lstm
        self.attn = base_model.attn
        self.norm = base_model.norm
        self.ffn = base_model.ffn
        self.dropout = base_model.dropout
        self.fc_event = base_model.fc_event
        self.fc_reg = base_model.fc_reg
        # Freeze base layers by default
        # FiLM modulators
        num_pitchers, emb_dim = self.pitcher_emb.weight.shape
        self.film_gamma = nn.Embedding(num_pitchers, hidden_dim)
        self.film_beta  = nn.Embedding(num_pitchers, hidden_dim)
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                             batch_first=True,
                             dropout=recurrent_dropout)
    def forward(self, x, pid):
        # x: [B, T, F]
        emb = self.embed_drop(self.pitcher_emb(pid))            # [B, emb_dim]
        emb_seq = emb.unsqueeze(1).repeat(1, x.size(1), 1)       # [B, T, emb_dim]
        x1 = torch.cat([x, emb_seq], dim=-1)                     # [B, T, F+emb]
        out1, _ = self.lstm1(x1)                                 # [B, T, H]
        # FiLM modulation
        gamma = self.film_gamma(pid).unsqueeze(1)                # [B, 1, H]
        beta  = self.film_beta(pid).unsqueeze(1)                 # [B, 1, H]
        mod1 = gamma * out1 + beta                               # [B, T, H]
        # Residual connection into LSTM2
        res2 = out1 + mod1                                       # [B, T, H]
        out2, _ = self.lstm2(res2)                               # [B, T, H]
        # Attention & heads
        attn_out, _ = self.attn(out2, out2, out2)                # [B, T, H]
        h = attn_out[:, -1, :]                                   # [B, H]
        h = self.norm(h)
        h = self.ffn(h)
        h = self.dropout(h)
        return self.fc_event(h), self.fc_reg(h)

def main(days: int = 30):
    """
    Fine-tune a 2-layer FiLM LSTM per pitcher using the frozen global backbone.
    """
    # 1) Locate frozen global checkpoint
    frozen_models = glob.glob('models/global_model_frozen*.pt')
    assert frozen_models, "No frozen global model found in models/"
    frozen_path = max(frozen_models, key=os.path.getmtime)
    log(f"Loading frozen global model from {frozen_path}")

    # 2) Load the checkpoint to inspect embedding size
    ckpt = torch.load(frozen_path, map_location=device)
    num_pitchers = ckpt['pitcher_emb.weight'].shape[0]
    log(f"Detected {num_pitchers} pitchers in global embedding")

    # 3) Reconstruct label‐encoder to get num_event_classes
    df_cache = load_cache()
    pitcher_df = get_pitchers_to_train(days)
    pitcher_ids = pitcher_df['mlbam_id'].astype(int).tolist()
    df_all = df_cache[df_cache['pitcher'].isin(pitcher_ids)]
    for tgt in REG_TARGETS:
        df_all[tgt] = df_all[tgt].fillna(0)
    _, le_event, _, _, _ = clean_and_scale_features(df_all)
    num_event_classes = len(le_event.classes_)

    # 4) Unpack the exact hyperparameters used for the frozen model
    #      (from best_cfg of global search)
    seq_len, batch_size, lr, hidden_dim, num_layers, \
    emb_dim, n_heads, attn_do, emb_do, ffn_exp, norm_type = (
        20, 128, 0.00011259939619198685,
        256, 1, 32, 1, 0.2, 0.1, 4, 'layer'
    )

    # 5) Instantiate the base model exactly
    base_model = GlobalMultiTaskAttentionLSTM(
        input_dim=len(FEATURES),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_event_classes=num_event_classes,
        dropout_rate=0.3,
        n_heads=n_heads,
        emb_dim=emb_dim,
        num_pitchers=num_pitchers,
        attn_dropout=attn_do,
        embed_dropout=emb_do,
        ffn_exp=ffn_exp,
        norm_type=norm_type
    ).to(device)

    # 6) Load its frozen weights (matching keys only)
    filtered = {k: v for k, v in ckpt.items() if k in base_model.state_dict()}
    base_model.load_state_dict(filtered, strict=False)
    base_model.eval()

    # 7) Wrap into FiLM + 2nd LSTM
    recurrent_dropout = 0.1
    model = TwoLayerFiLMLSTM(base_model, hidden_dim, recurrent_dropout).to(device)

    # 8) Freeze all but film modulators & lstm2
    for name, p in model.named_parameters():
        if not (name.startswith('film_') or name.startswith('lstm2')):
            p.requires_grad = False

    # 9) Build PID→embed‐index map
    pitcher_to_idx = {pid: i for i, pid in enumerate(pitcher_ids)}

    metrics = []
    for pid in pitcher_ids:
        name = pitcher_df.loc[pitcher_df['mlbam_id'] == pid, 'pitcher_name'].iloc[0]
        log(f"==== FiLM fine-tuning for {name} (ID={pid}) ====")

        emb_idx = pitcher_to_idx[pid]
        # Subset and scale this pitcher's data
        df_raw = df_cache[df_cache['pitcher'] == pid].copy()
        if df_raw.empty:
            log("  No data, skipping.")
            continue
        for tgt in REG_TARGETS:
            df_raw[tgt] = df_raw[tgt].fillna(0)
        df_clean, _, _, scaler_x, scaler_y = clean_and_scale_features(df_raw)

        # Split 90/10
        split = int(len(df_clean) * 0.9)
        df_train, df_val = df_clean.iloc[:split], df_clean.iloc[split:]
        if len(df_train) < 1 or len(df_val) < 1:
            log("  Not enough data, skipping.")
            continue

        # Compute class weights over full set of events
        y_vals = df_train[TARGET_EVENT].astype(int).values
        counts = np.bincount(y_vals, minlength=num_event_classes)
        weights = len(y_vals) / (counts.astype(float) + 1e-8)
        weights[counts == 0] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        crit_event = nn.CrossEntropyLoss(weight=weight_tensor)
        crit_reg   = nn.MSELoss()

        # Datasets & loaders
        train_ds = PitcherDataset(df_train, seq_len)
        val_ds   = PitcherDataset(df_val, seq_len)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE_FINE, shuffle=True)
        val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE_FINE, shuffle=False)

        # Optimizer & scheduler
        opt = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=INITIAL_LR)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                     factor=0.5, patience=PATIENCE_FINE)

        # Fine-tune with early stopping
        best_loss, no_imp = float('inf'), 0
        for ep in range(1, EPOCHS_FINE + 1):
            model.train()
            for Xb, ye, yr in train_ld:
                Xb, ye, yr = Xb.to(device), ye.to(device), yr.to(device)
                opt.zero_grad()
                idx_t = torch.full((len(Xb),), emb_idx, dtype=torch.long, device=device)
                logits, regs = model(Xb, idx_t)
                loss = crit_event(logits, ye) + crit_reg(regs, yr)
                loss.backward()
                opt.step()

            # Validate (inverse‐transform regs)
            model.eval()
            val_losses, preds, targs, regs_list, targs_list = [], [], [], [], []
            with torch.no_grad():
                for Xv, yv, rv in val_ld:
                    Xv, yv, rv = Xv.to(device), yv.to(device), rv.to(device)
                    idx_t = torch.full((len(Xv),), emb_idx,
                                       dtype=torch.long, device=device)
                    lo, re = model(Xv, idx_t)
                    cl = crit_event(lo, yv).item()
                    r_np, t_np = re.cpu().numpy(), rv.cpu().numpy()
                    regs_list.extend(r_np)
                    targs_list.extend(t_np)
                    
                    orig_r = scaler_y.inverse_transform(r_np)
                    orig_t = scaler_y.inverse_transform(t_np)
                    rl = np.mean((orig_r - orig_t) ** 2)
                    val_losses.append(cl + rl)
                    preds.extend(torch.argmax(lo,1).cpu().numpy())
                    targs.extend(yv.cpu().numpy())
            orig_regs  = scaler_y.inverse_transform(np.array(regs_list))
            orig_targs = scaler_y.inverse_transform(np.array(targs_list))
            val_loss = np.mean(val_losses)
            sched.step(val_loss)
            log(f"Ep {ep}/{EPOCHS_FINE} – val_loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss, best_state, no_imp = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE_FINE:
                    log("  Early stopping")
                    break

        # Load best, eval final metrics
        model.load_state_dict(best_state)
        event_acc = accuracy_score(targs, preds)
        regs_all = scaler_y.inverse_transform(np.array(regs_list))
        targs_all= scaler_y.inverse_transform(np.array(targs_list))
        reg_mse   = mean_squared_error(targs_all, regs_all)
        log(f"Final – Acc: {event_acc:.4f}, MSE: {reg_mse:.4f}")

        # Save model & record metrics
        out_p = f"models/individual_{name}_2layerFILM{dating}.pt"
        torch.save(model.state_dict(), out_p)
        log(f"Saved to {out_p}")
        metrics.append({
            'pitcher_id': pid, 'pitcher_name': name,
            'val_loss': best_loss,
            'accuracy': event_acc,
            'reg_mse': reg_mse
        })

    # 10) Write metrics CSV
    dfm = pd.DataFrame(metrics)
    csv_out = f"fine_tune_metrics_2layerFILM{dating}.csv"
    dfm.to_csv(csv_out, index=False)
    log(f"Saved metrics to {csv_out}")
    log("=== 2-layer FiLM fine-tuning complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    args = parser.parse_args()
    main(args.days)
