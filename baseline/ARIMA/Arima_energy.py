import pandas as pd
import numpy as np
import json
from pathlib import Path
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据
df = pd.read_csv("data_clean/energy.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.rename(columns={'timestamp': 'ds', 'y': 'y'})

# 拆分数据（80% 训练，20% 测试）
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 拟合 ARIMA 模型（自动参数搜索）
model = auto_arima(train['y'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
forecast = model.predict(n_periods=len(test))
y_pred = pd.Series(forecast, index=test['ds'])

# 自动处理 NaN：删除预测中所有 NaN 项，并同步删除真实值对应项
y_true = test.set_index('ds')['y']
if y_pred.isna().any():
    print("⚠️ NaN detected in predictions. Dropping NaN entries for evaluation.")
    y_true = y_true[~y_pred.isna()]
    y_pred = y_pred.dropna()

# 若无有效预测，输出错误信息并保存失败记录
if len(y_pred) == 0 or len(y_true) == 0:
    print("❌ ARIMA failed to generate valid predictions for energy dataset.")
    metrics = {
        "energy": {
            "arima": {
                "mae": None,
                "rmse": None,
                "smape": None,
                "note": "ARIMA failed: no valid predictions (all NaN)"
            }
        }
    }
    output_path = Path("baseline_results/ARIMA/Arima_energy_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    exit()

# 评估指标
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print("[Energy – ARIMA]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

# 保存结果
metrics = {
    "energy": {
        "arima": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

output_path = Path("baseline_results/ARIMA/Arima_energy_metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)
