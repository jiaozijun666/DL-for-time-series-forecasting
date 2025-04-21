import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path
import json

# 读取数据
df = pd.read_csv("data_clean/gait.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 排序并设置索引
df = df.sort_values('timestamp').drop_duplicates(subset='timestamp')
df = df.set_index('timestamp')

# 自动频率推断
try:
    inferred_freq = pd.infer_freq(df.index)
    df = df.asfreq(inferred_freq)
except:
    inferred_freq = None

# 删除NaN
df = df.dropna()

# 拆分训练和测试
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

try:
    # 拟合ARIMA
    model = pm.auto_arima(train['y'], seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=True)
    y_pred = model.predict(n_periods=len(test))
    y_pred = pd.Series(y_pred, index=test.index)

    # 删除NaN预测值
    mask = ~y_pred.isna()
    y_pred = y_pred[mask]
    y_true = test['y'][mask]

    # 若全部为NaN，报错
    if len(y_true) == 0:
        raise ValueError("All predictions are NaN")

    # 评估
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    metrics = {
        "gait": {
            "arima": {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "smape": round(smape, 2)
            }
        }
    }

except Exception as e:
    metrics = {
        "gait": {
            "arima": {
                "mae": None,
                "rmse": None,
                "smape": None,
                "note": f"ARIMA failed: {str(e)}"
            }
        }
    }

# 保存
output_path = Path("baseline_results/ARIMA/Arima_gait_metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(metrics)
