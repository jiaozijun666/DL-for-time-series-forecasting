import pandas as pd
from pathlib import Path

# === Step 1: 读取原始 gait 数据 ===
df_raw = pd.read_csv("/Users/jzj/Desktop/DL-for-time-series-forecasting/datasets/gait_raw.csv")  # 原始数据路径（你可以改成绝对路径）

# === Step 2: 假设为等间隔采样 ===
# 假设采样频率为 100Hz（即每 10 毫秒采样一次）
sampling_rate_hz = 100
interval_ms = int(1000 / sampling_rate_hz)  # 10ms

# 创建 timestamp 时间戳列，从一个固定起始时间生成
df_raw['timestamp'] = pd.date_range(
    start='2020-01-01 00:00:00',
    periods=len(df_raw),
    freq=f'{interval_ms}ms'
)

# === Step 3: 重新命名列为 ['timestamp', 'y'] ===
# 假设你想保留第一列作为目标变量 y
df_clean = df_raw.rename(columns={df_raw.columns[0]: 'y'})[['timestamp', 'y']]

# === Step 4: 保存为清洗后的版本 ===
output_path = Path("/Users/jzj/Desktop/DL-for-time-series-forecasting/data_clean/gait.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(output_path, index=False)

print("✅ 清理后的 gait.csv 已保存至 data_clean/")

