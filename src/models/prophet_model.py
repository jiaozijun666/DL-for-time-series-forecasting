import pandas as pd
import numpy as np
import logging
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("models.prophet_model")

class TimeSeriesForecaster:
    def __init__(self, freq='D', test_size=0.2):
        self.freq = freq
        self.test_size = test_size
        self.best_params = None
        self.model = None

    def train(self, series):
        try:
            self.best_params = self._tune_parameters(series)
            self.model = self._build_model(self.best_params)

            t = pd.date_range(start='2000-01-01', periods=len(series), freq=self.freq)
            df = pd.DataFrame({'ds': t, 'y': series})

            split_index = int(len(df) * (1 - self.test_size))
            train_df = df.iloc[:split_index]

            self.model.fit(train_df)

        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            raise

    def predict(self, series):
        t = pd.date_range(start='2000-01-01', periods=len(series), freq=self.freq)
        df = pd.DataFrame({'ds': t, 'y': series})

        split_index = int(len(df) * (1 - self.test_size))
        test_df = df.iloc[split_index:]
        future = test_df[['ds']].copy()

        forecast = self.model.predict(future)

        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        return y_true, y_pred

    def _tune_parameters(self, series):
        # 可调参网格
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'fourier_order': [5, 10]
        }

        t = pd.date_range(start='2000-01-01', periods=len(series), freq=self.freq)
        df = pd.DataFrame({'ds': t, 'y': series})
        split_index = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:split_index]
        val_df = df.iloc[split_index:]

        best_score = float('inf')
        best_params = None

        for cps in param_grid['changepoint_prior_scale']:
            for sps in param_grid['seasonality_prior_scale']:
                for sm in param_grid['seasonality_mode']:
                    for fo in param_grid['fourier_order']:
                        try:
                            model = Prophet(
                                changepoint_prior_scale=cps,
                                seasonality_prior_scale=sps,
                                seasonality_mode=sm
                            )
                            model.add_seasonality(name='custom', period=7, fourier_order=fo)
                            model.fit(train_df)
                            future = val_df[['ds']].copy()
                            forecast = model.predict(future)
                            y_true = val_df['y'].values
                            y_pred = forecast['yhat'].values
                            rmse = mean_squared_error(y_true, y_pred, squared=False)

                            if rmse < best_score:
                                best_score = rmse
                                best_params = {
                                    'changepoint_prior_scale': cps,
                                    'seasonality_prior_scale': sps,
                                    'seasonality_mode': sm,
                                    'fourier_order': fo
                                }
                        except Exception as e:
                            logger.warning(f"跳过无效参数组合: {e}")

        if not best_params:
            logger.warning("未找到最优参数，使用默认值")
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'fourier_order': 10
            }

        logger.info(f"最优参数: {best_params}")
        return best_params

    def _build_model(self, params):
        if not params:
            raise ValueError("无法构建模型，参数为 None")

        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        model.add_seasonality(name='custom', period=7, fourier_order=params['fourier_order'])
        return model
