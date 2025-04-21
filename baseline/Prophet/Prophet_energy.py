import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json
from pathlib import Path

# 读取数据
df = pd.read_csv('data_clean/energy.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.rename(columns={'timestamp': 'ds', 'y': 'y'})  # Prophet 要求列名为 ds 和 y

# 拆分训练集和测试集
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# 拟合 Prophet 模型
model = Prophet()
model.fit(train)

# 生成预测
future = model.make_future_dataframe(periods=len(test), freq=None)
forecast = model.predict(future)

# 匹配预测值（更稳健）
df_result = test.merge(forecast[['ds', 'yhat']], on='ds', how='left')
df_result = df_result.dropna(subset=['yhat'])

y_true = df_result['y']
y_pred = df_result['yhat']

# 评估指标
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

print("[Energy – Prophet]")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"SMAPE: {smape:.2f}%")

# 保存结果
metrics = {
    "energy": {
        "prophet": {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "smape": round(smape, 2)
        }
    }
}

output_path = Path("baseline_results/Prophet/Prophet_energy_metrics.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)