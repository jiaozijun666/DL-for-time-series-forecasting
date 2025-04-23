import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from models.lgbm_model import run_lgbm_forecast
from utils.metrics import evaluate_all_metrics

# ✅ 数据集路径
datasets = {
    "air_quality": "data/data_clean/air_quality.csv",
    "energy": "data/data_clean/energy.csv",
    "gait": "data/data_clean/gait.csv",
    "metro": "data/data_clean/metro.csv",
    "productivity": "data/data_clean/productivity.csv"
}

# ✅ 输出目录
os.makedirs("results/metric_csv/LGBM", exist_ok=True)
os.makedirs("results/prediction_png/LGBM", exist_ok=True)
os.makedirs("results/loss_png/LGBM", exist_ok=True)
os.makedirs("results/comparisons_png/LGBM", exist_ok=True)

# ✅ 统一目标列
target_column = "y"

# ✅ 主循环
def run_all():
    for name, path in datasets.items():
        print(f"\n🚀 Running LGBM for {name}...")

        df = pd.read_csv(path)
        series = df[target_column].dropna().values

        start = time.time()
        y_true, y_pred, evals_result = run_lgbm_forecast(series)
        runtime = round(time.time() - start, 3)

        # ✅ 保存评估指标
        metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
        pd.DataFrame([metrics]).to_csv(f"results/metric_csv/LGBM/metrics_{name}.csv", index=False)

        # ✅ True vs Predicted 图
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label='True', linewidth=2)
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f"LGBM Prediction - {name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/prediction_png/LGBM/prediction_{name}.png")
        plt.close()

        # training loss VS validation loss
        plt.figure(figsize=(10, 4))
        plt.plot(evals_result['training']['l1'], label='Training Loss', linewidth=2)
        plt.plot(evals_result['valid_1']['l1'], label='Validation Loss', linestyle='--')
        plt.title(f"LGBM Loss - {name}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"results/loss_png/LGBM/loss_{name}.png")
        plt.close()

            # ✅ 生成三张指标对比柱状图（整体汇总）
    all_metrics = []
    for name in datasets.keys():
        metric_path = f"results/metric_csv/LGBM/metrics_{name}.csv"
        if os.path.exists(metric_path):
            df = pd.read_csv(metric_path)
            all_metrics.append({
                "Dataset": name,
                "MAE": df.loc[0, "MAE"],
                "RMSE": df.loc[0, "RMSE"],
                "MAPE": df.loc[0, "MAPE"]
            })
    
    summary_df = pd.DataFrame(all_metrics)

    for metric in ["MAE", "RMSE", "MAPE"]:
        plt.figure(figsize=(8, 5))
        bars = plt.bar(summary_df["Dataset"], summary_df[metric])
        for bar, value in zip(bars, summary_df[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
        plt.title(f"LGBM {metric} per datasets")
        plt.ylabel(metric)
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(f"results/comparisons_png/LGBM/comparison_{metric}.png")
        plt.close()

    
        print(f"✅ Finished {name} | Runtime: {runtime}s")

if __name__ == "__main__":
    run_all()
