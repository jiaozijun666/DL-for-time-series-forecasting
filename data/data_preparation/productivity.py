import pandas as pd
from pathlib import Path

RAW = Path("datasets/productivity_raw.csv")
OUT = Path("data_clean"); OUT.mkdir(exist_ok=True)

df = pd.read_csv(RAW)

# 日期标准化
df['timestamp'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# 去除缺失 & 重命名目标列
df_clean = df[['timestamp', 'actual_productivity']].rename(columns={'actual_productivity': 'y'})
df_clean = df_clean.dropna()

# 保存
df_clean.to_csv(OUT / 'productivity.csv', index=False)
print("✅ Saved to: data_clean/productivity.csv")
