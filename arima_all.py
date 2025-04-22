import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from models.arima_model import run_auto_arima_forecast
from utils.metrics import evaluate_all_metrics

# 数据集路径
datasets = {
    "air_quality": "datasets/data_clean/air_quality.csv",
    "energy": "datasets/data_clean/energy.csv",
    "gait": "datasets/data_clean/gait.csv",
    "metro": "datasets/data_clean/metro.csv",
    "productivity": "datasets/data_clean/productivity.csv"
}

# 每个数据集的季节参数设定
seasonal_params = {
    "air_quality": (True, 24),
    "energy": (False, 1),
    "gait": (False, 1),
    "metro": (True, 24),
    "productivity": (False, 1)
}

# 输出目录
os.makedirs("results/AutoARIMA", exist_ok=True)

for name, path in datasets.items():
    print(f"🔁 Running auto_arima for {name}...")

    df = pd.read_csv(path)
    df = df.dropna()
    if 'y' not in df.columns:
        raise ValueError(f"{name} missing 'y' column")

    seasonal, m = seasonal_params[name]

    start = time.time()
    try:
        y_true, y_pred = run_auto_arima_forecast(df, seasonal=seasonal, m=m)
        runtime = round(time.time() - start, 3)

        # 指标评估
        metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
        pd.DataFrame([metrics]).to_csv(f"results/AutoARIMA/metrics_{name}.csv", index=False)

        # 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label='True')
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f"Auto ARIMA Forecast - {name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/AutoARIMA/prediction_{name}.png")
        plt.close()

        print(f"✅ Done: {name} | Runtime: {runtime}s")
    except Exception as e:
        print(f"❌ Failed: {name} | Reason: {str(e)}")

print("🎉 All datasets processed with AutoARIMA.")
