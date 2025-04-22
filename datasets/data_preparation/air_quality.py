import pandas as pd
from pathlib import Path

# 设定路径
RAW = Path('datasets/air_quality_raw.csv')       # 原始数据文件
OUT = Path('data_clean'); OUT.mkdir(exist_ok=True)

# 读取 CSV：使用分号作为分隔符，小数用逗号，防止乱码
df = pd.read_csv(RAW, sep=";", decimal=",", engine='python')

# 去除空白列，清理列名空格
df.columns = df.columns.str.strip()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 合并 Date + Time 成 timestamp
df['timestamp'] = pd.to_datetime(
    df['Date'] + ' ' + df['Time'],
    format='%d/%m/%Y %H.%M.%S',
    errors='coerce'
)

# 选择目标列（CO 浓度），重命名为 y
df_clean = df[['timestamp', 'CO(GT)']].rename(columns={'CO(GT)': 'y'})

# 删除无效数据（空值与 -200）
df_clean = df_clean.dropna()
df_clean = df_clean[df_clean['y'] > -100]

# 保存为干净格式
df_clean.to_csv(OUT / 'air_quality.csv', index=False)
print('✅ Saved to: data_clean/air_quality.csv')
