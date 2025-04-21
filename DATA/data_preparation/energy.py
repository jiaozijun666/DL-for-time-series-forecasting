import pandas as pd
from pathlib import Path

RAW = Path('datasets/energy_raw.csv')          # 原始文件
OUT = Path('data_clean'); OUT.mkdir(exist_ok=True)

df = pd.read_csv(RAW)

# 时间列标准化
df['timestamp'] = pd.to_datetime(df['date'])

# 选择目标列 Appliances，重命名为 y
df_clean = df[['timestamp', 'Appliances']].rename(columns={'Appliances': 'y'})

# 保存为标准格式
df_clean.to_csv(OUT / 'energy.csv', index=False)
print("✅ Saved to: data_clean/energy.csv")
