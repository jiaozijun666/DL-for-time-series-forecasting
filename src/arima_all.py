import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from models.arima_model import run_auto_arima_forecast
from utils.metrics import evaluate_all_metrics

# ✅ 数据集路径
datasets = {
    "air_quality": "datasets/data_clean/air_quality.csv",
    "energy": "datasets/data_clean/energy.csv",
    "gait": "datasets/data_clean/gait.csv",
    "metro": "datasets/data_clean/metro.csv",
    "productivity": "datasets/data_clean/productivity.csv"
}

# ✅ 每个数据集的季节性设置和是否降采样
seasonal_params = {
    "air_quality":   {"seasonal": True,  "m": 24,  "downsample": False},
    "energy":        {"seasonal": False, "m": 1,   "downsample": True},
    "gait":          {"seasonal": False, "m": 1,   "downsample": True},
    "metro":         {"seasonal": True,  "m": 24,  "downsample": False},
    "productivity":  {"seasonal": False, "m": 1,   "downsample": False}
}

# ✅ 输出文件夹
os.makedirs("results/ARIMA", exist_ok=True)

# ✅ 循环处理每个数据集
for name, path in datasets.items():
    print(f"🔁 Running Auto ARIMA for {name}...")

    df = pd.read_csv(path)
    if 'y' not in df.columns:
        print(f"❌ Skipped {name}: missing column 'y'")
        continue

    df = df.dropna().reset_index(drop=True)

    config = seasonal_params[name]

    try:
        start = time.time()

        # ✅ 执行模型
        y_true, y_pred = run_auto_arima_forecast(
            df,
            seasonal=config["seasonal"],
            m=config["m"],
            downsample=config["downsample"],
            max_order=5, max_p=2, max_q=2, max_P=1, max_Q=1
        )

        runtime = round(time.time() - start, 3)

        # ✅ 评估指标
        metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
        pd.DataFrame([metrics]).to_csv(f"results/ARIMA/metrics_{name}.csv", index=False)

        # ✅ 绘图
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="True", linewidth=2)
        plt.plot(y_pred, label="Predicted", linestyle='--')
        plt.title(f"ARIMA Forecast - {name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/ARIMA/prediction_{name}.png")
        plt.close()

        print(f"✅ Finished {name} | Runtime: {runtime}s\n")

    except Exception as e:
        print(f"❌ Failed {name}: {str(e)}")


