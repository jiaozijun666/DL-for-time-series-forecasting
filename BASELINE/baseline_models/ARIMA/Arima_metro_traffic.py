import pandas as pd
import numpy as np
import json
from pathlib import Path
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据
df = pd.read_csv('DATA/data_clean/metro.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.rename(columns={'timestamp': 'ds', 'y': 'y'})
df = df.dropna()

# 拆分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

try:
    # 限制搜索空间以避免爆内存
    model = auto_arima(
        train['y'],
        seasonal=True,
        m=24,
        start_p=1, start_q=1,
        max_p=2, max_q=2,
        start_P=0, start_Q=0,
        max_P=1, max_Q=1,
        max_order=5,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )

    # 预测
    y_pred = model.predict(n_periods=len(test))
    y_pred = pd.Series(y_pred).dropna().reset_index(drop=True)
    y_true = test['y'].reset_index(drop=True)

    # 对齐
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if len(y_true) == 0:
        raise ValueError("All predictions are NaN")

    # 计算误差
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    print("[Metro – ARIMA]")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"SMAPE: {smape:.2f}%")

    metrics = {
        "metro": {
            "arima": {
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "smape": round(smape, 2)
            }
        }
    }

except Exception as e:
    metrics = {
        "metro": {
            "arima": {
                "mae": None,
                "rmse": None,
                "smape": None,
                "note": f"ARIMA failed: {str(e)}"
            }
        }
    }

# 保存结果
output_path = Path("BASELINE/baseline_results/ARIMA/Arima_metro_metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)
