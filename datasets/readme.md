# 1.solar energy（能源）
来源：https://github.com/laiguokun/multivariate-time-series-data/tree/master/solar-energy
转csv的代码
```python
import pandas as pd
import gzip, pathlib, shutil

# 路径
src_gz  = pathlib.Path("/Users/jzj/Downloads/solar_AL.txt.gz")   # ← 你的实际路径
dst_csv = pathlib.Path("/Users/jzj/Downloads/solar_AL.csv")

# 1️⃣ 直接用 pandas 读 gzip → DataFrame
df = pd.read_csv(src_gz, 
                 compression="gzip", 
                 header=None)          # 原文件无列名

# 2️⃣ 设置列名（根据官方说明：第一列时间索引 + 其他137条序列）
# 如果你的文件只有单站点，通常第一列是时间戳，其余一列是功率
if df.shape[1] == 2:
    df.columns = ["datetime", "power"]
else:                                  # 多站点/多变量示例
    df.columns = ["datetime"] + [f"series_{i}" for i in range(1, df.shape[1])]
    
# 3️⃣ 转换时间格式并设为索引（可选）
df["datetime"] = pd.to_datetime(df["datetime"])
df.set_index("datetime", inplace=True)

# 4️⃣ 保存为 CSV
df.to_csv(dst_csv)
print("✅ Saved:", dst_csv, "shape:", df.shape)
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



# 2. air quality（环境）
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
# 3.exchange rate（金融领域）
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


# 4. metro interstate traffic volume(交通领域)
### 下载链接：https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume  解压后直接是csv
### gpt代码雏形（LSTM Seq2Seq）
```python
import torch, torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, in_dim, hid, out_horizon):
        super().__init__()
        self.encoder = nn.LSTM(in_dim, hid, batch_first=True)
        self.decoder = nn.LSTM(1, hid, batch_first=True)
        self.fc      = nn.Linear(hid, 1)
        self.H       = out_horizon

    def forward(self, x):
        # x: (B,L,d)
        _, (h,c) = self.encoder(x)
        dec_in = torch.zeros(x.size(0), 1, 1, device=x.device)  # first decoder input
        outs=[]
        for _ in range(self.H):
            out, (h,c) = self.decoder(dec_in, (h,c))
            pred = self.fc(out)          # (B,1,1)
            outs.append(pred.squeeze(1))
            dec_in = pred
        return torch.cat(outs, dim=1)    # (B,H)

# 训练时：X(L×d)标准化；y 反标准化再计算指标
```

# 5. gait（生物医疗）
### 下载链接： https://archive.ics.uci.edu/dataset/760/multivariate+gait+data
### gpt建议：
#### （1）清理：
```python
import pandas as pd
df = pd.read_csv("gait_raw/joint_angle_data.csv")

# 保证 frame 0‑100 共 101 帧
df = df[df["frame"].between(0,100)]        

# 可选：去掉极端角度
df = df[(df.angle>-120) & (df.angle<120)]

#——保存统一版
df.to_csv("data/gait_clean.csv", index=False)
```
#### （2）构造数据集 & DataLoader
```python
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader

SEQ_IN, SEQ_OUT = 51, 50               # 输入 / 输出长度
FEATURES = ["angle"]                   # 单变量；若想多关节多腿可展开

class GaitDataset(Dataset):
    def __init__(self, df, split="train"):
        # leave‐one‐subject‐out；亦可 random split
        test_ids  = [9,10]
        if split=="train": self.df = df[~df.subject.isin(test_ids)]
        else:              self.df = df[df.subject.isin(test_ids)]

        # 分组生成 (101, 1) 的 ndarray
        self.series = []
        for _, g in self.df.groupby(["subject","condition","rep","leg","joint"]):
            seq = g.sort_values("frame")["angle"].to_numpy(dtype=np.float32)
            if len(seq)==101:
                self.series.append(seq)

    def __len__(self):  return len(self.series)
    def __getitem__(self, idx):
        seq = self.series[idx]
        x = seq[:SEQ_IN]
        y = seq[SEQ_IN:]
        return torch.from_numpy(x).unsqueeze(-1), torch.from_numpy(y).unsqueeze(-1)

train_loader = DataLoader(GaitDataset(df,"train"), batch_size=128, shuffle=True, drop_last=True)
test_loader  = DataLoader(GaitDataset(df,"test"),  batch_size=128, shuffle=False)
```
#### (3)基线模型
```python
# Naïve / Persistence
def persistence_mae(test_loader):
    import torch, numpy as np
    maes=[]
    for x,y in test_loader:
        y_hat = x[:,-1:,:].repeat(1,SEQ_OUT,1)  # 把最后一帧当预测
        maes.append(torch.abs(y_hat-y).mean().item())
    print("Persistence MAE:", np.mean(maes))

#线性外推 + Spline（scikit‑learn）
from sklearn.linear_model import Ridge
def ridge_baseline(train_loader, test_loader):
    X_tr, y_tr = [], []
    for x,y in train_loader:
        X_tr.append(x.squeeze(-1)); y_tr.append(y.squeeze(-1))
    X_tr = torch.cat(X_tr).numpy(); y_tr = torch.cat(y_tr).numpy()

    model = Ridge(alpha=1.0).fit(X_tr, y_tr)      # 多输出回归
    mae=[]
    for x,y in test_loader:
        pred = model.predict(x.squeeze(-1).numpy())
        mae.append(np.abs(pred - y.squeeze(-1).numpy()).mean())
    print("Ridge MAE:", np.mean(mae))
```

#### (4) deep learning
```python
# TCN（Temporal Convolutional Network）
import torch.nn as nn
class Chomp1d(nn.Module):
    def __init__(self, chomp):  super().__init__(); self.chomp=chomp
    def forward(self,x):        return x[:,:,:-self.chomp]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs,n_outputs,k=3,d=1,drop=0.2):
        super().__init__()
        pad = (k-1)*d
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs,n_outputs,k,padding=pad,dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(n_outputs,n_outputs,k,padding=pad,dilation=d),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(drop)
        )
        self.down = nn.Conv1d(n_inputs,n_outputs,1) if n_inputs!=n_outputs else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.net(x)
        return self.relu(out + self.down(x))

class TCN(nn.Module):
    def __init__(self,n_inputs,n_outputs,channel_sizes=[32,32,32],k=3):
        super().__init__()
        layers=[]
        for i,c in enumerate(channel_sizes):
            d = 2**i
            layers.append(TemporalBlock(n_inputs if i==0 else channel_sizes[i-1],
                                        c,k,d))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(channel_sizes[-1], n_outputs)

    def forward(self,x):             # x: (B, L, C)
        y = self.tcn(x.transpose(1,2)).transpose(1,2)  # (B,L,Ch)
        return self.fc(y)            # (B,L_out,1)

# 训练循环
import torch, torch.nn.functional as F, pytorch_lightning as pl
class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = TCN(1, 1, [32,32,32])
    def forward(self,x):  return self.net(x)
    def training_step(self,batch,idx):
        x,y = batch
        pred = self(x)
        loss = F.l1_loss(pred, y)       # MAE
        self.log("train_mae", loss)
        return loss
    def validation_step(self,batch,idx):
        x,y = batch
        pred = self(x)
        mae  = F.l1_loss(pred,y)
        self.log("val_mae", mae, prog_bar=True)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e‑4)

model = LitModel()
trainer = pl.Trainer(max_epochs=40, accelerator="gpu", devices=1)
trainer.fit(model, train_loader, test_loader)

# LSTM/GRU baseline
class Seq2SeqRNN(nn.Module):
    def __init__(self,h=64,layers=2):
        super().__init__()
        self.enc = nn.GRU(1,h,layers,batch_first=True)
        self.dec = nn.GRU(1,h,layers,batch_first=True)
        self.fc  = nn.Linear(h,1)
    def forward(self,x):
        B = x.size(0)
        h,_ = self.enc(x)                    # 用最后一个隐状态
        context = h[:,-1:,:]                 # (B,1,H)
        dec_in  = x[:,-1:,:].repeat(1,SEQ_OUT,1)   # teacher forcing=0
        out,_ = self.dec(dec_in, context.permute(1,0,2))
        return self.fc(out)
```

#### (5)评估指标 & 可视化
```python
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt, seaborn as sns

# 对每个 (subject, cond, rep) 画真实/预测角度曲线
def plot_sample(model, ds, idx=0):
    x,y = ds[idx]
    with torch.no_grad():
        pred = model(x.unsqueeze(0)).squeeze(0).squeeze(-1).numpy()
    plt.figure(figsize=(7,3))
    plt.plot(range(101), np.concatenate([x.squeeze(), y.squeeze()]), label="Truth")
    plt.plot(range(51,101), pred, '--', label="Pred")
    plt.axvline(50, c='k', ls=':')
    plt.legend(); plt.title("Angle forecast demo")

# 统计误差
def full_eval(model, loader):
    maes=[]
    with torch.no_grad():
        for x,y in loader:
            pred = model(x)
            maes.append(torch.abs(pred-y).mean().item())
    print("Global MAE:", np.mean(maes))
```






# 6. electricity（跟solar energy属于同一个领域了，建议使用solar）
### 初始data太大了，在这个链接里： https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
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
