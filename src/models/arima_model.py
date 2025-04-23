import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_auto_arima_forecast(
    df, 
    seasonal=False, 
    m=1, 
    test_size=0.2,
    max_order=5, 
    max_p=2, max_q=2,
    max_P=1, max_Q=1,
    downsample=False
):
    """
    稳定版 Auto ARIMA 模型（支持限制阶数 + 降采样）

    参数：
        - df: 包含 'y' 列的 DataFrame
        - seasonal: 是否使用季节性模型
        - m: 季节周期（如每日 = 24，10分钟 = 144）
        - test_size: 测试集比例
        - max_order, max_p, max_q, max_P, max_Q: 限制阶数，避免爆炸
        - downsample: 是否进行降采样（每隔一点取一点）

    返回：
        - y_test: 测试集真实值
        - y_pred: 预测值
    """
    df = df.dropna().reset_index(drop=True)

    if 'y' not in df.columns:
        raise ValueError("DataFrame must contain column 'y'.")

    if downsample:
        df = df.iloc[::2].reset_index(drop=True)

    y = df['y'].values
    n = len(y)
    train_size = int(n * (1 - test_size))
    y_train, y_test = y[:train_size], y[train_size:]

    model = auto_arima(
        y_train,
        seasonal=seasonal,
        m=m,
        max_order=max_order,
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q,
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    y_pred = model.predict(n_periods=len(y_test))

    # 去除NaN对齐长度
    y_pred = pd.Series(y_pred).dropna().reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    min_len = min(len(y_test), len(y_pred))

    return y_test[:min_len], y_pred[:min_len]
