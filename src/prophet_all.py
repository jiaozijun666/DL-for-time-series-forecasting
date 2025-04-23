import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from models.prophet_model import run_prophet_forecast
from utils.metrics import evaluate_all_metrics

# ✅ 数据集路径（根据你项目的 clean 版本）
datasets = {
    "air_quality": "datasets/data_clean/air_quality.csv",
    "energy": "datasets/data_clean/energy.csv",
    "gait": "datasets/data_clean/gait.csv",
    "metro": "datasets/data_clean/metro.csv",
    "productivity": "datasets/data_clean/productivity.csv"
}

# ✅ 输出文件夹
os.makedirs("results/Prophet", exist_ok=True)

# ✅ 统一预测目标列
target_column = "y"

# ✅ 每个数据集的频率（手动设置）
freqs = {
    "air_quality": "H",       # 每小时
    "energy": "10min",        # 每10分钟
    "gait": "10ms",           # 每10毫秒（Prophet 不支持这么高频，建议 downsample）
    "metro": "H",             # 每小时
    "productivity": "D"       # 每天
}

# ✅ 循环执行
for name, path in datasets.items():
    print(f"🔮 Running Prophet for {name}...")

    df = pd.read_csv(path)
    series = df[target_column].dropna().values

    # 获取频率
    freq = freqs[name]

    # 模型运行
    start = time.time()
    try:
        y_true, y_pred = run_prophet_forecast(series, freq=freq)
        runtime = round(time.time() - start, 3)

        # 评估 + 保存
        metrics = evaluate_all_metrics(y_true, y_pred, threshold=100, runtime=runtime)
        pd.DataFrame([metrics]).to_csv(f"results/Prophet/metrics_{name}.csv", index=False)

        # 可视化
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label='True', linewidth=2)
        plt.plot(y_pred, label='Predicted', linestyle='--')
        plt.title(f"Prophet Prediction - {name}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/Prophet/prediction_{name}.png")
        plt.close()

        print(f"✅ Finished: {name} | Runtime: {runtime}s\n")
    except Exception as e:
        print(f"❌ Error running Prophet for {name}: {e}\n")

print("🎉 All datasets completed with Prophet.")
