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
log_filename = f"individual_pitcher_train{dating}.txt"
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
# Load model definitions
from global_pitcher_model_search import GlobalMultiTaskAttentionLSTM, freeze_global_model

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

# Detect latest frozen model
frozen_models = glob.glob('models/global_model_frozen*.pt')
assert frozen_models, "No frozen global model found in models/"
frozen_path = max(frozen_models, key=os.path.getmtime)
log(f"Loading frozen global model from {frozen_path}")

# Load cache & pitcher list
df_cache = load_cache()
pitcher_df = get_pitchers_to_train()
metrics = []

for _, row in pitcher_df.iterrows():
    pid = int(row['mlbam_id'])
    name = row['pitcher_name']
    log(f"==== Fine-tuning for {name} (ID={pid}) ====")
    # Subset raw
    df_raw = df_cache[df_cache['pitcher']==pid].copy()
    if df_raw.empty:
        log("  No data, skipping.")
        continue
    for tgt in REG_TARGETS:
        df_raw[tgt] = df_raw[tgt].fillna(0)
    df_clean, le_event, encs, scaler_x, scaler_y = clean_and_scale_features(df_raw)

    # Split 90/10
    split = int(len(df_clean)*0.9)
    df_train = df_clean.iloc[:split]
    df_val   = df_clean.iloc[split:]
    if len(df_train)<1 or len(df_val)<1:
        log("  Not enough data, skipping.")
        continue

    # — per-pitcher class weights —
    y_vals = df_train[TARGET_EVENT].values.astype(int)
    cnts   = np.bincount(y_vals, minlength=len(le_event.classes_))
    weights= len(y_vals)/(cnts.astype(float)+1e-8)
    weights[cnts==0] = 0.0
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    crit_event = nn.CrossEntropyLoss(weight=weight_tensor)
    crit_reg   = nn.MSELoss()

    # Build datasets
    seq_len=5  # as per global search
    train_ds=PitcherDataset(df_train, seq_len)
    val_ds  =PitcherDataset(df_val, seq_len)
    train_ld=DataLoader(train_ds, batch_size=BATCH_SIZE_FINE, shuffle=True)
    val_ld  =DataLoader(val_ds, batch_size=BATCH_SIZE_FINE, shuffle=False)

    # Construct model
    best_cfg=(5,32,0.0009221,64,1,16,1,0.2,0.0,4,'layer')
    (_,_,_,hidden,nl,emb_dim,n_heads,attn_do,emb_do,ffn_exp,norm_type) = best_cfg
    model=GlobalMultiTaskAttentionLSTM(input_dim=len(FEATURES), emb_dim=emb_dim,
                                       num_pitchers=1, hidden_dim=hidden,
                                       num_layers=nl, num_event_classes=len(le_event.classes_),
                                       dropout_rate=0.3, n_heads=n_heads,
                                       attn_dropout=attn_do, embed_dropout=emb_do,
                                       ffn_exp=ffn_exp, norm_type=norm_type)
    model=model.to(device)

    # Load and filter weights
    state=torch.load(frozen_path, map_location=device)
    state={k:v for k,v in state.items() if not(k.startswith('pitcher_emb') or k.startswith('fc_event') or k.startswith('fc_reg'))}
    model.load_state_dict(state, strict=False)

    # Freeze + unfreeze for ample data
    model=freeze_global_model(model)
    if len(df_raw)>50:
        for nm,param in model.named_parameters():
            if nm.startswith('attn') or nm.startswith('ffn'): param.requires_grad=True
        log("  Unfroze attention & FFN layers for ample data.")

    # Optimizer & scheduler
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(trainable, lr=INITIAL_LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=PATIENCE_FINE)

    # Fine-tune
    best_loss=float('inf'); patience=0
    for ep in range(1, EPOCHS_FINE+1):
        model.train()
        for Xb, ye, yr in train_ld:
            Xb, ye, yr = Xb.to(device), ye.to(device), yr.to(device)
            opt.zero_grad()
            lo, re = model(Xb, torch.zeros(len(Xb), dtype=torch.long, device=device))
            loss = crit_event(lo, ye) + crit_reg(re, yr)
            loss.backward()
            opt.step()
        # Validation
        model.eval(); val_losses=[]; all_preds=[]; all_tgts=[]; all_regs=[]; all_rtg=[]
        with torch.no_grad():
            for Xv, yv, rv in val_ld:
                Xv, yv, rv = Xv.to(device), yv.to(device), rv.to(device)
                lo, re = model(Xv, torch.zeros(len(Xv), dtype=torch.long, device=device))
                val_losses.append((crit_event(lo, yv)+crit_reg(re, rv)).item())
                all_preds.extend(torch.argmax(lo,1).cpu().numpy())
                all_tgts.extend(yv.cpu().numpy())
                all_regs.extend(re.cpu().numpy())
                all_rtg.extend(rv.cpu().numpy())
        avg_val=np.mean(val_losses)
        sched.step(avg_val)
        log(f"Epoch {ep}/{EPOCHS_FINE} - val_loss: {avg_val:.4f}")
        if avg_val<best_loss:
            best_loss, best_state = avg_val, copy.deepcopy(model.state_dict()); patience=0
        else:
            patience+=1
            if patience>=PATIENCE_FINE: log("  Early stopping"); break

    # Load best state
    model.load_state_dict(best_state)

    # Final metrics with inverse-transform
    orig_re = scaler_y.inverse_transform(np.array(all_regs))
    orig_rt = scaler_y.inverse_transform(np.array(all_rtg))
    event_acc = accuracy_score(all_tgts, all_preds)
    reg_mse   = mean_squared_error(orig_rt, orig_re)
    log(f"Final metrics - Acc: {event_acc:.4f}, Reg MSE: {reg_mse:.4f}")

    # Save model & metrics
    out_path = f"models/individual_{name}{dating}.pt"
    torch.save(model.state_dict(), out_path)
    log(f"Saved model to {out_path}")
    metrics.append({
        'pitcher_id': pid, 'pitcher_name': name,
        'val_loss': best_loss, 'accuracy': event_acc, 'reg_mse': reg_mse
    })

# Write metrics
dfm = pd.DataFrame(metrics)
csv_out = f"fine_tune_metrics{dating}.csv"
dfm.to_csv(csv_out, index=False)
log(f"Saved metrics to {csv_out}")
log("Individual fine-tuning complete.")
