import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from pathlib import Path

# 读取数据
df = pd.read_csv('data_clean/gait.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.rename(columns={'timestamp': 'ds', 'y': 'y'})

# 去除 NaN
df = df.dropna(subset=['ds', 'y'])

# 拆分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ARIMA建模
try:
    model = auto_arima(train['y'], seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    y_pred = model.predict(n_periods=len(test))

    # 转换为 pandas Series 并对齐时间戳
    y_pred = pd.Series(y_pred, index=test['ds'])

    # 清理预测结果中的 NaN（如果有）
    if y_pred.isna().all():
        raise ValueError("All predictions are NaN")

    y_true = test.set_index('ds')['y']
    valid_idx = y_pred.dropna().index.intersection(y_true.dropna().index)

    y_true = y_true.loc[valid_idx]
    y_pred = y_pred.loc[valid_idx]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

    print("[Gait – ARIMA]")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"SMAPE: {smape:.2f}%")

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
    print(f"ARIMA failed: {e}")
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

# 保存结果
output_path = Path("baseline_results/ARIMA/Arima_gait_metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)
