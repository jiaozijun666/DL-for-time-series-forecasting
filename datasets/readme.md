## 1. electricity的data太大了，在这个链接里： https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
### zip解压后使用代码 df = pd.read_csv('LD2011_2014.txt', sep=';', index_col=0, parse_dates=True, decimal=',')


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


