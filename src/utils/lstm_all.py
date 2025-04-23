# === utils/lstm_all.py ===
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.lstm_model import run_lstm_forecast
from utils.metrics import evaluate_all_metrics

# ✅ 数据集路径
datasets = {
    "air_quality": "datasets/data_clean/air_quality.csv",
    "energy": "datasets/data_clean/energy.csv",
    "gait": "datasets/data_clean/gait.csv",
    "metro": "datasets/data_clean/metro.csv",
    "productivity": "datasets/data_clean/productivity.csv"
}

# ✅ LSTM 训练参数
params = {
    'HIDDEN_SIZE': 128,
    'N_LAYERS': 3,
    'DROPOUT': 0.2,
    'LEARNING_RATE': 1e-4,
    'EPOCHS': 60,
    'PATIENCE': 10,
    'DEVICE': "cuda" if torch.cuda.is_available() else "cpu"
}

# ✅ 滞后特征处理函数
LOOKBACK, HORIZON = 24, 24

def make_supervised_data(series, lookback=24, horizon=24):
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)

# ✅ 创建输出目录
os.makedirs("results/LSTM", exist_ok=True)

# ✅ 主流程循环
for name, path in datasets.items():
    print(f"🚀 Running LSTM for {name}...")
    df = pd.read_csv(path)
    series = df['y'].dropna().values.astype(np.float32)
    X, y = make_supervised_data(series, LOOKBACK, HORIZON)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    start = time.time()
    y_true, y_pred, val_losses, mae, rmse = run_lstm_forecast(X_train, y_train, X_test, y_test, {**params, 'HORIZON': HORIZON})
    runtime = round(time.time() - start, 3)

    # ✅ 评估并保存指标
    metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
    pd.DataFrame([metrics]).to_csv(f"results/LSTM/metrics_{name}.csv", index=False)

    # ✅ 预测图
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:, 0], label='True', linewidth=2)
    plt.plot(y_pred[:, 0], label='Predicted', linestyle='--')
    plt.title(f"LSTM Prediction - {name} (t+1)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/LSTM/prediction_{name}.png")
    plt.close()

    # ✅ 验证损失图
    plt.figure(figsize=(6, 3))
    plt.plot(val_losses, label="Val Loss", color='orange')
    plt.title(f"Validation Loss - {name}")
    plt.tight_layout()
    plt.savefig(f"results/LSTM/loss_{name}.png")
    plt.close()
