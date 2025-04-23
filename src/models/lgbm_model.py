import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def generate_lag_features(series, window_size=12):
    """
    将时间序列转换为带滞后特征的监督学习格式。
    输入：series（一维 np.array 或 pd.Series），window_size（滞后步数）
    输出：DataFrame，包含目标 y 和 lag_1, lag_2, ..., lag_window_size 特征列
    """
    df = pd.DataFrame({'y': series})
    for i in range(1, window_size + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    return df

def run_lgbm_forecast(series, window_size=12, test_size=0.2, random_state=42):
    """
    对一个时间序列使用 LightGBM 回归建模预测。
    输入：
        - series: 一维时间序列（np.array 或 pd.Series）
        - window_size: 滞后特征窗口大小
        - test_size: 测试集比例
    输出：
        - y_test: 测试集真实值
        - y_pred: 模型预测值
    """
    df = generate_lag_features(series, window_size)
    X = df.drop(columns=['y']).values
    y = df['y'].values

    # 顺序分割（保持时间一致性）
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # LightGBM 回归建模
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return y_test, y_pred
