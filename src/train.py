import argparse
import os
import shutil
import pandas as pd
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default="data.yaml")
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载预训练模型
    model = YOLO(args.model)

    # 开始训练
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz
    )

    # ✅ 保存 best.pt 到 SageMaker 模型目录
    best_path = "runs/detect/train/weights/best.pt"
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(best_path):
        shutil.copy(best_path, os.path.join(model_dir, "best.pt"))
        print("✅ best.pt 已保存到 /opt/ml/model/")
    else:
        print("⚠️ 没找到 best.pt，请检查训练输出路径")

    # ✅ 验证模型并保存指标
    metrics = model.val()
    results = {
        "Precision": [metrics.box.p],
        "Recall": [metrics.box.r],
        "mAP@0.5": [metrics.box.map50],
        "mAP@0.5:0.95": [metrics.box.map]
    }
    metrics_path = os.path.join(model_dir, "final_metrics.csv")
    pd.DataFrame(results).to_csv(metrics_path, index=False)
    print("✅ final_metrics.csv 已保存到 /opt/ml/model/")

if __name__ == "__main__":
    main()
