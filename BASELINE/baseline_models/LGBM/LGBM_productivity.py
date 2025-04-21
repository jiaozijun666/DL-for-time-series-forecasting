import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
from pathlib import Path

# === 1. 加载数据 ===
data_path = Path("data_clean/productivity.csv")
df = pd.read_csv(data_path, parse_dates=["timestamp"])

# === 2. 时间特征 ===
df["weekday"] = df["timestamp"].dt.weekday
df["month"] = df["timestamp"].dt.month

# === 3. 滞后特征（1~30）===
for lag in [1, 2, 3, 7, 30]:
    df[f"lag_{lag}"] = df["y"].shift(lag)

df = df.dropna().sort_values("timestamp")

# === 4. 数据切分 ===
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

features = [col for col in df.columns if col.startswith("lag_") or col in ["weekday", "month"]]

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

# === 7. 输出 & 保存 ===
print(f"[Productivity - LightGBM]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

results = {
    "productivity": {
        "lightgbm": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

Path("results").mkdir(exist_ok=True)
with open("baseline_results/LGBM/LGBM_productivity_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
