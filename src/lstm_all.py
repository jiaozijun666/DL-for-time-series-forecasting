import os
import time
import glob
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models.lstm_model import run_lstm_forecast
from utils.metrics import evaluate_all_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ 数据集路径
datasets = {
    "air_quality": "data/data_clean/air_quality.csv",
    "energy": "data/data_clean/energy.csv",
    "gait": "data/data_clean/gait.csv",
    "metro": "data/data_clean/metro.csv",
    "productivity": "data/data_clean/productivity.csv"
}

# ✅ LSTM 训练参数
params = {
    'HIDDEN_SIZE': 128,
    'N_LAYERS': 3,
    'DROPOUT': 0.2,
    'LEARNING_RATE': 1e-4,
    'EPOCHS': 60,
    'PATIENCE': 10,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu",
    'BATCH_SIZE': 128
}

LOOKBACK, HORIZON = 24, 24

class WindowDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def load_split_scale(path, target='y', time_col='timestamp'):
    df = pd.read_csv(path)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
    else:
        df = df.reset_index().rename(columns={"index": "seq_idx"})
        time_col = "seq_idx"
    df = df.reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in (time_col, target)]
    if not feat_cols:
        feat_cols = [target]

    n = len(df)
    train_end, val_end = int(n * 0.7), int(n * 0.85)
    train_df, val_df, test_df = df[:train_end], df[train_end:val_end], df[val_end:]

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    return train_df, val_df, test_df, feat_cols

def make_windows(df, feat_cols, target, lookback, horizon):
    X, y = [], []
    feats = df[feat_cols].astype(np.float32).values
    targets = df[target].astype(np.float32).values
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feats[i:i+lookback])
        y.append(targets[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)


for name, path in datasets.items():
    print(f"🚀 Running LSTM for {name}...")
    start = time.time()
    df_tr, df_va, df_te, feat_cols = load_split_scale(path)

    X_tr, y_tr = make_windows(df_tr, feat_cols, 'y', LOOKBACK, HORIZON)
    X_va, y_va = make_windows(df_va, feat_cols, 'y', LOOKBACK, HORIZON)
    X_te, y_te = make_windows(df_te, feat_cols, 'y', LOOKBACK, HORIZON)

    train_loader = DataLoader(WindowDS(X_tr, y_tr), batch_size=params['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(WindowDS(X_va, y_va), batch_size=params['BATCH_SIZE'], shuffle=False)
    test_loader = DataLoader(WindowDS(X_te, y_te), batch_size=params['BATCH_SIZE'], shuffle=False)

    loaders = (train_loader, val_loader, test_loader)
    y_true, y_pred, train_losses, val_losses, mae, rmse = run_lstm_forecast(
        loaders, len(feat_cols), {**params, 'HORIZON': HORIZON}, return_train_loss=True
    )
    runtime = round(time.time() - start, 3)

    metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
    pd.DataFrame([metrics]).to_csv(f"results/metric_csv/LSTM/metrics_{name}.csv", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:200, 0], label='True', linewidth=2)
    plt.plot(y_pred[:200, 0], label='Pred', color='orange')
    plt.title(f"{name} – step t+1")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"results/prediction_png/LSTM/prediction_{name}.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Val Loss", color='orange')
    plt.title(f"Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/loss_png/LSTM/loss_{name}.png")
    plt.close()

# 绘制comparision_bar

# === 配置路径 ===
input_dir = "results/metric_csv/LSTM"
output_dir = "results/comparisions_png/LSTM"
os.makedirs(output_dir, exist_ok=True)

# === 读取所有指标 CSV ===
all_files = glob.glob(os.path.join(input_dir, "metrics_*.csv"))
records = []
for file in all_files:
    df = pd.read_csv(file)
    name = os.path.splitext(os.path.basename(file))[0].replace("metrics_", "")
    df.columns = [col.upper() for col in df.columns]
    df.insert(0, "DATASET", name)
    records.append(df)

metrics_df = pd.concat(records).reset_index(drop=True)

# === 画图函数（美化版） ===
def plot_bar(metric_name):
    plt.figure(figsize=(9, 6))
    values = metrics_df[["DATASET", metric_name]].sort_values(metric_name)
    bars = plt.bar(values["DATASET"], values[metric_name], color="#4C72B0", edgecolor='black')
    for bar, val in zip(bars, values[metric_name]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01 * val, f"{val:.3f}",
                 ha='center', va='bottom', fontsize=10, fontweight='medium')
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f"{metric_name} per dataset", fontsize=16, weight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparision_{metric_name}.png")
    plt.close()

# === 仅绘制这四个指标图 ===
metrics_to_plot = [
    "MAE",
    "RMSE",
    "MAPE",
    "SMAPE"
]

for metric in metrics_to_plot:
    if metric in metrics_df.columns:
        plot_bar(metric)
    else:
        print(f"⚠️ Metric '{metric}' not found in CSV files.")
