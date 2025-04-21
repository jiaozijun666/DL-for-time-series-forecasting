import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path

# === 1. 读取数据 ===
df = pd.read_csv("data_clean/air_quality.csv", parse_dates=["timestamp"])
df = df.rename(columns={"timestamp": "ds", "y": "y"})

df = df.sort_values("ds")
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

# === 2. 拟合 Prophet + 自定义 daily seasonality ===
model = Prophet()
model.add_seasonality(name='daily', period=24, fourier_order=5)
model.fit(df_train)

# === 3. 预测测试集时间段 ===
future = df_test[["ds"]].copy()
forecast = model.predict(future)

# === 4. 评估 ===
y_true = df_test["y"].values
y_pred = forecast["yhat"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

print(f"[Air Quality - Prophet Improved]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

results = {
    "air_quality": {
        "prophet": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

Path("baseline_results").mkdir(exist_ok=True)
with open("baseline_results/Prophet_air_quality_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
