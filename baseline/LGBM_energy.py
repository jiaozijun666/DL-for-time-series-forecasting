import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
from pathlib import Path

# === 1. 加载数据 ===
data_path = Path("data_clean/energy.csv")
df = pd.read_csv(data_path, parse_dates=["timestamp"])

# === 2. 时间特征 ===
df["hour"] = df["timestamp"].dt.hour
df["weekday"] = df["timestamp"].dt.weekday

# === 3. 滞后特征（1–144）===
for lag in [1, 2, 3, 6, 12, 24, 144]:
    df[f"lag_{lag}"] = df["y"].shift(lag)

df = df.dropna().sort_values("timestamp")

# === 4. 数据切分 ===
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

features = [col for col in df.columns if col.startswith("lag_") or col in ["hour", "weekday"]]

X_train, y_train = df_train[features], df_train["y"]
X_val, y_val = df_val[features], df_val["y"]
X_test, y_test = df_test[features], df_test["y"]

# === 5. 模型训练 ===
model = lgb.LGBMRegressor(n_estimators=200)
model.fit(X_train, y_train)

# === 6. 预测与评估 ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
smape = 100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)))

# === 7. 输出 & 保存结果 ===
print(f"[Energy - LightGBM]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

results = {
    "energy": {
        "lightgbm": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

Path("results").mkdir(exist_ok=True)
with open("baseline_results/energy_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

