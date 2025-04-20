import pandas as pd
from pathlib import Path

RAW = Path("datasets/gait_raw.csv")
OUT = Path("data_clean"); OUT.mkdir(exist_ok=True)

df = pd.read_csv(RAW)

# 过滤单个被试、单关节、单腿、单次实验
subset = df.query("subject == 1 and condition == 1 and replication == 1 and leg == 1 and joint == 1").copy()

# 假设每步时间间隔为 20ms = 0.02s
start = pd.Timestamp("2020-01-01 00:00:00")
subset['timestamp'] = start + pd.to_timedelta(subset['time'] * 20, unit='ms')

# 只保留 timestamp 与角度值
df_clean = subset[['timestamp', 'angle']].rename(columns={'angle': 'y'})

# 保存
df_clean.to_csv(OUT / 'gait.csv', index=False)
print("✅ Saved to: data_clean/gait.csv")
