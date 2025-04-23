import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_auto_arima_forecast(df, seasonal=False, m=1, test_size=0.2):
    """
    通用 ARIMA 自动建模函数，适配任意单变量时间序列。
    
    参数：
        - df: 包含 'y' 列的 DataFrame（时间索引可选）
        - seasonal: 是否拟合季节性
        - m: 季节周期（如24表示24小时周期）
        - test_size: 测试集比例（如 0.2 表示 20% 测试）
    
    返回：
        - y_test: 真实值
        - y_pred: 预测值
    """
    df = df.dropna().reset_index(drop=True)
    y = df['y'].values

    n = len(y)
    train_size = int(n * (1 - test_size))
    y_train, y_test = y[:train_size], y[train_size:]

    model = auto_arima(
        y_train,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )

    y_pred = model.predict(n_periods=len(y_test))

    # 去掉 NaN 并对齐长度
    y_pred = pd.Series(y_pred).dropna().reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    min_len = min(len(y_test), len(y_pred))
    return y_test[:min_len], y_pred[:min_len]
