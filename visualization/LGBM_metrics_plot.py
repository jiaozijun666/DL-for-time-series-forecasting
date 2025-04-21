import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# === 1. 文件映射 ===
file_map = {
    "air_quality_metrics.json": "air_quality",
    "energy_metrics.json": "energy",
    "gait_metrics.json": "gait",
    "metro_traffic_metrics.json": "metro",
    "productivity_metrics.json": "productivity"
}

records = []

# === 2. 读取每个 JSON 文件 ===
for file_name, dataset in file_map.items():
    file_path = Path("baseline_results") / file_name
    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)
            metrics = data[dataset]["lightgbm"]
            metrics["dataset"] = dataset
            records.append(metrics)

# === 3. 整合为 DataFrame ===
df = pd.DataFrame(records).set_index("dataset")
print("📊 Baseline Results:")
print(df)

# === 4. 画图函数 ===
def plot_metric(metric):
    plt.figure(figsize=(9, 5))
    
    # 柱状图
    ax = df[metric].plot(kind="bar", color="cornflowerblue", edgecolor="black")

    # 设置 y 轴最大值（自动缩放 + 手动上限）
    y_max = df[metric].max()
    plt.ylim(0, y_max * 1.15)

    # 添加数值标签
    for i, val in enumerate(df[metric]):
        ax.text(i, val + y_max * 0.02, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # 标题和样式
    plt.title(f"LightGBM {metric.upper()} across Datasets")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=30)
    plt.grid(axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = Path("visualization") / f"lgbm_{metric}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path)
    print(f"✅ Saved: {out_path}")


# === 5. 输出 MAE / RMSE / SMAPE 图 ===
for metric in ["mae", "rmse", "smape"]:
    plot_metric(metric)
