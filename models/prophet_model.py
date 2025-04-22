import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_prophet_forecast(series, freq='D', test_size=0.2):
    """
    使用 Facebook Prophet 对单变量时间序列进行建模和预测。

    输入：
        - series: 1D np.array 或 pd.Series
        - freq: 时间频率（'D', 'H', '10min' 等）
        - test_size: 测试集比例（默认 20%）

    输出：
        - y_test: 测试集真实值
        - y_pred: Prophet 模型预测值
    """
    n = len(series)
    t = pd.date_range(start='2000-01-01', periods=n, freq=freq)
    df = pd.DataFrame({'ds': t, 'y': series})

    # 拆分训练集和测试集
    split_index = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # 建模
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    # 构造未来时间点
    future = df[['ds']].iloc[split_index:].reset_index(drop=True)
    forecast = model.predict(future)

    # 预测值和真实值
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values

    return y_true, y_pred
