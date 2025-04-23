# ✅ lgbm_model.py（支持训练+验证loss记录 + evals_result 输出）
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def generate_lag_features(series, window_size=12):
    df = pd.DataFrame({'y': series})
    for i in range(1, window_size + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    return df


def run_lgbm_forecast(series, window_size=12, test_size=0.2, random_state=42):
    """
    使用 LightGBM 模型拟合单变量时间序列（滑动窗口 + 验证loss记录）
    """
    df = generate_lag_features(series, window_size)
    X = df.drop(columns=['y']).values
    y = df['y'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)

    evals_result = {}

    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=random_state
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric='l1',
        callbacks=[lgb.record_evaluation(evals_result)]
    )

    y_pred = model.predict(X_val)
    return y_val, y_pred, evals_result