from prophet import Prophet
import pandas as pd

def run_prophet_forecast(series, freq='D', test_size=0.2, seasonality_mode='additive'):
    n = len(series)
    t = pd.date_range(start='2000-01-01', periods=n, freq=freq)
    df = pd.DataFrame({'ds': t, 'y': series})

    split_idx = int(n * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    model = Prophet(seasonality_mode=seasonality_mode)
    model.fit(train_df)

    future = test_df[['ds']].reset_index(drop=True)
    forecast = model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values

    return y_true, y_pred
