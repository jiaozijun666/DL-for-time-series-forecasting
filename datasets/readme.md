## 1. electricity的data太大了，在这个链接里： https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
### zip解压后使用代码
```python
df = pd.read_csv('LD2011_2014.txt', sep=';', index_col=0, parse_dates=True, decimal=',')
```
这里是原数据集（过大）


### 上传的electricity_sample.csv文件是after data cleaning的，缩小了范围，chatgpt说清洗后的数据不影响最后模型结果
#### 缩小范围的过程
```python
import pandas as pd

# 原始文件路径
df = pd.read_csv("LD2011_2014.txt", sep=";", index_col=0, parse_dates=True, decimal=",")

# 确保列名格式正确
df.columns = [col.strip() for col in df.columns]

# 选择时间段（比如：2013 年全年）
df_small = df.loc["2013-01-01":"2013-12-31"]

# 选取前 20 个用户（或你感兴趣的一些区域）
df_small = df_small.iloc[:, :20]

# 保存为轻量文件
df_small.to_csv("electricity_sample.csv")
```



## 2. air quality的数据使用方法：(chatgpt写的，没有测试）
### 📄 数据概述
- **来源**：[UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **时间范围**：2004 年 3 月 – 2005 年 4 月（分钟级别）
- **数据文件**：`AirQualityUCI.csv`
- **缺失值标记**：`-200`

#### 📌 特征说明
| 变量 | 含义 |
|------|------|
| `CO(GT)` | 一氧化碳浓度（mg/m³）✅ 可选目标变量 |
| `NOx(GT)` / `NO2(GT)` | 氮氧化物浓度（ppb / µg/m³）✅ 可选目标变量 |
| `PT08.S1` ~ `PT08.S5` | 多个传感器输出值 |
| `T`, `RH`, `AH` | 温度、相对湿度、绝对湿度 |
| `Date`, `Time` | 时间戳列（需合并）

---

### 🧪 使用流程

#### Step 1: 加载数据

```python
import pandas as pd

df = pd.read_csv(
    "AirQualityUCI.csv",
    sep=";",
    decimal=",",
    parse_dates=[["Date", "Time"]],
    infer_datetime_format=True
)
df = df.iloc[:, :-2]  # 移除空列
```

#### Step 2: 设置时间索引
```python
df.columns = [col.strip().replace(" ", "_") for col in df.columns]
df.rename(columns={"Date_Time": "datetime"}, inplace=True)
df.set_index("datetime", inplace=True)
```

#### Step 3: 清洗缺失值
```python
df.replace(-200, pd.NA, inplace=True)
df = df.interpolate(method='time')
df = df.dropna()
```

#### Step 4: 特征选择
```python
target = "NO2_GT"
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]
```

#### Step 5: 构造滑动窗口数据（适用于 LSTM、Transformer 等）
```python
def create_sequences(X, y, window=24):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X.iloc[i:(i + window)].values)
        ys.append(y.iloc[i + window])
    return np.array(Xs), np.array(ys)

Xs, ys = create_sequences(X, y, window=24)
```
## 3.exchange rate
来源：github https://github.com/laiguokun/multivariate-time-series-data/tree/master/exchange_rate
Mac无法解压这里面的文件
上传的exchange_rate_with_date.csv是通过下面的代码解出来的
```python
import pandas as pd
from datetime import timedelta, datetime

# 读取
df = pd.read_csv("exchange_rate.txt.gz", compression="gzip", header=None)
df.columns = [
    "Australia", "UK", "Canada", 
    "Switzerland", "China", "Japan", 
    "New_Zealand", "Singapore"
]

# 构造日期索引（从1990-01-01开始，逐日）
start_date = datetime(1990, 1, 1)
df["date"] = [start_date + timedelta(days=i) for i in range(len(df))]
df.set_index("date", inplace=True)

# 保存为 CSV
df.to_csv("datasets/exchange/exchange_rate_with_date.csv")

print("✅ exchange_rate_with_date.csv saved with synthetic date index.")
```
## 4.solar energy
来源：https://github.com/laiguokun/multivariate-time-series-data/tree/master/solar-energy
转csv的代码
```python
import pandas as pd
from datetime import timedelta, datetime

# 读取
df = pd.read_csv("/Users/jzj/Downloads/exchange_rate.txt.gz", compression="gzip", header=None)
df.columns = [
    "Australia", "UK", "Canada", 
    "Switzerland", "China", "Japan", 
    "New_Zealand", "Singapore"
]

# 构造日期索引（从1990-01-01开始，逐日）
start_date = datetime(1990, 1, 1)
df["date"] = [start_date + timedelta(days=i) for i in range(len(df))]
df.set_index("date", inplace=True)

# 保存为 CSV
df.to_csv("/Users/jzj/Downloads/exchange_rate_with_date.csv")

print("✅ exchange_rate_with_date.csv saved with synthetic date index.")
```
上传的文件已经经过了除0处理（源文件很多一整排都是0的）
清理代码：
```python
#把文件里一整排都是0的行删除
import pandas as pd
import numpy as np      

# 读取 CSV 文件
df = pd.read_csv("/Users/jzj/Downloads/solar_AL_cleaned.csv", header=0, parse_dates=["datetime"], index_col="datetime")
# 检查缺失值
print("缺失值统计：")
print(df.isnull().sum())
# 删除全为0的行
df = df[(df != 0).any(axis=1)]
# 保存为新的 CSV 文件
df.to_csv("/Users/jzj/Downloads/solar_AL_cleaned_no_zeros.csv")
print("✅ Cleaned data without zeros saved:", "/Users/jzj/Downloads/solar_AL_cleaned_no_zeros.csv", "shape:", df.shape)
```
这个代码是我本地copilot跑出来的，大家要换路径
