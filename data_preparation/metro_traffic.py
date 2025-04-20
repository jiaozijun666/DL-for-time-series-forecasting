import pandas as pd
from pathlib import Path

RAW = Path("datasets/metro_traffic_volume_raw.csv")
OUT = Path("data_clean"); OUT.mkdir(exist_ok=True)

df = pd.read_csv(RAW)

# 时间列标准化
df['timestamp'] = pd.to_datetime(df['date_time'])

# 目标列重命名
df_clean = df[['timestamp', 'traffic_volume']].rename(columns={'traffic_volume': 'y'})

# 保存为标准格式
df_clean.to_csv(OUT / 'metro.csv', index=False)
print("✅ Saved to: data_clean/metro.csv")
