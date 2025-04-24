# ✅ prophet_all.py（完整版，配合 TimeSeriesForecaster）
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from models.prophet_model import TimeSeriesForecaster
from utils.metrics import evaluate_all_metrics

# ✅ 数据路径与频率设定
datasets = {
    "air_quality": "data/data_clean/air_quality.csv",
    "energy": "data/data_clean/energy.csv",
    "gait": "data/data_clean/gait.csv",
    "metro": "data/data_clean/metro.csv",
    "productivity": "data/data_clean/productivity.csv"
}
freqs = {
    "air_quality": "h",
    "energy": "10min",
    "gait": "10ms",
    "metro": "h",
    "productivity": "D"
}

# ✅ 输出目录结构
os.makedirs("results/metric_csv/Prophet", exist_ok=True)
os.makedirs("results/prediction_png/Prophet", exist_ok=True)
os.makedirs("results/comparisons_png/Prophet", exist_ok=True)

target_column = "y"

def run_all():
    all_metrics = []

    for name, path in datasets.items():
        print(f"\n🔮 Running Prophet for {name}...")
        df = pd.read_csv(path)
        if target_column not in df.columns:
            print(f"❌ Skipped {name}: missing column '{target_column}'")
            continue

        series = df[target_column].dropna().values
        freq = freqs[name]

        try:
            model = TimeSeriesForecaster(freq=freq, test_size=0.2)
            model.train(series)
            y_true, y_pred = model.predict(series)
            runtime = round(time.time(), 3)

            # ✅ 保存指标
            metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
            pd.DataFrame([metrics]).to_csv(f"results/metric_csv/Prophet/metrics_{name}.csv", index=False)
            all_metrics.append({"Dataset": name, "MAE": metrics['MAE'], "RMSE": metrics['RMSE'], "MAPE": metrics['MAPE']})

            # ✅ 预测图
            plt.figure(figsize=(10, 4))
            plt.plot(y_true, label="True", linewidth=2)
            plt.plot(y_pred, label="Predicted", linestyle='--')
            plt.title(f"Prophet Prediction - {name}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/prediction_png/Prophet/prediction_{name}.png")
            plt.close()

            

        except Exception as e:
            print(f"❌ Failed {name}: {str(e)}")

    # ✅ 总体比较图
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        for metric in ["MAE", "RMSE", "MAPE"]:
            plt.figure(figsize=(8, 5))
            bars = plt.bar(summary_df["Dataset"], summary_df[metric])
            for bar, value in zip(bars, summary_df[metric]):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
            plt.title(f"Prophet {metric} Comparison across Datasets")
            plt.ylabel(metric)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(f"results/comparisons_png/Prophet/Comparison_{metric}.png")
            plt.close()

if __name__ == "__main__":
    run_all()
