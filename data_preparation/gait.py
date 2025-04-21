import pandas as pd

# 读取原始 gait 数据
df = pd.read_csv("datasets/gait_raw.csv")

# 只取 angle 作为 y，假设它是等间隔采样的
y = df["angle"].reset_index(drop=True)

# 构造等间隔 timestamp（假设为 0.01s 间隔）
timestamp = pd.date_range(start="2020-01-01", periods=len(y), freq="10ms")

# 构建清洗后的 DataFrame
clean_df = pd.DataFrame({
    "timestamp": timestamp,
    "y": y
})

# 保存到 data_clean 文件夹
clean_df.to_csv("data_clean/gait.csv", index=False)