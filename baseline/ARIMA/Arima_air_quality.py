import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path
import warnings
import pmdarima as pm

warnings.filterwarnings("ignore")

# === 1. 加载数据 ===
df = pd.read_csv("data_clean/air_quality.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# 只使用 y
y = df["y"].values

# === 2. 划分 train / test ===
n = len(y)
train_end = int(n * 0.85)
y_train = y[:train_end]
y_test = y[train_end:]

# === 3. 自动拟合 SARIMA 模型 ===
model = pm.auto_arima(
    y_train,
    seasonal=True,
    m=24,                    # 每天 24 小时，空气质量通常有日周期性
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    max_d=2, max_D=1,
    stepwise=True,          # 启用逐步搜索，避免暴力穷举
    suppress_warnings=True,
    error_action='ignore',
    trace=True               # 可选：查看每一步的模型尝试
)


# === 4. 预测 ===
y_pred = model.predict(n_periods=len(y_test))

# === 5. 评估 ===
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))

# === 6. 输出 ===
print(f"[Air Quality - ARIMA]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

results = {
    "air_quality": {
        "arima": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

Path("baseline_results").mkdir(exist_ok=True)
with open("baseline_results/Arima/Arima_air_quality_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

