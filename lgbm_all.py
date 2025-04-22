import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# ✅ 从 models 模块中导入 LGBM 模型函数
from models.lgbm_model import run_lgbm_forecast

# ✅ 从 utils 模块中导入评估指标函数
from utils.metrics import evaluate_all_metrics

# ✅ 数据集路径（根据你的上传结构）
datasets = {
    "air_quality": "datasets/data_clean/air_quality.csv",
    "energy": "datasets/data_clean/energy.csv",
    "gait": "datasets/data_clean/gait.csv",
    "metro": "datasets/data_clean/metro.csv",
    "productivity": "datasets/data_clean/productivity.csv"
}

# ✅ 创建输出目录
os.makedirs("results/LGBM", exist_ok=True)

# ✅ 目标列名称（统一为 'y'）
target_column = "y"

# ✅ 主循环处理所有数据集
for name, path in datasets.items():
    print(f"🚀 Running LGBM for {name}...")

    # 读取数据并提取目标列
    df = pd.read_csv(path)
    series = df[target_column].dropna().values

    # 模型训练 & 预测
    start = time.time()
    y_true, y_pred = run_lgbm_forecast(series)
    runtime = round(time.time() - start, 3)

    # 评估指标
    metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
    pd.DataFrame([metrics]).to_csv(f"results/LGBM/metrics_{name}.csv", index=False)

    # 可视化预测结果
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='True', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"LGBM Prediction - {name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/LGBM/prediction_{name}.png")
    plt.close()
