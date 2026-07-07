# 🎯 YOLOv8n Face Detection on AWS SageMaker

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8n-Ultralytics-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/AWS-SageMaker-orange?style=flat-square&logo=amazonaws" />
  <img src="https://img.shields.io/badge/Amazon-S3-green?style=flat-square&logo=amazons3" />
  <img src="https://img.shields.io/badge/Python-3.10-yellow?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square" />
</p>

<p align="center">
  <b>English</b> | <a href="#chinese">中文说明</a>
</p>

---

## 📖 Overview

This project implements a **lightweight face detection system** using **YOLOv8n** trained on a subset of the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) dataset, with the full training pipeline hosted on **AWS SageMaker** and data managed via **Amazon S3**.

The best model (50 Epochs) achieves:

| Metric | Score |
|--------|-------|
| Precision | **0.834** |
| Recall | 0.413 |
| mAP@0.5 | **0.498** |
| mAP@0.5:0.95 | 0.262 |

---

## 🏗️ System Architecture

```
WIDER Face Dataset
       │
       ▼
Amazon S3 (Dataset Storage)
       │
       ▼
Amazon SageMaker (Training Platform)
       │
       ▼
  YOLOv8n Model
       │
  ┌────┴────┐
  ▼         ▼
best.pt  final_metrics.csv
  │         │
  └────┬────┘
       ▼
Amazon S3 (Results Archive)
```

---

## 📁 Repository Structure

```
YOLO-Face-Detection/
├── src/
│   └── train.py                  # SageMaker training entry script
├── Dateset
│   ├── /WIDER_subset/WIDER_train
│   └── /WIDER_subset/WIDER_val
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment

### Software

| Item | Version |
|------|---------|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.0 |
| Ultralytics | 8.2.0 |
| OpenCV | 4.8 |
| SageMaker SDK | Latest |

### Hardware (AWS SageMaker)

| Item | Config |
|------|--------|
| Instance | ml.g5.xlarge |
| GPU | NVIDIA A10G |
| VRAM | 24 GB |
| CPU | 4 vCPU |
| RAM | 16 GB |

---

## 🚀 Quick Start

### Step 1 — Install Dependencies

```bash
pip install ultralytics==8.2.0
pip install torch torchvision
pip install opencv-python matplotlib
```

### Step 2 — Download WIDER Face Dataset

```python
import os
import urllib.request

urls = {
    "WIDER_train.zip":      "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip",
    "WIDER_val.zip":        "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip",
    "wider_face_split.zip": "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"
}

save_dir = "C:/your/path/YOLO/"
os.makedirs(save_dir, exist_ok=True)

for name, url in urls.items():
    print("Downloading:", name)
    urllib.request.urlretrieve(url, os.path.join(save_dir, name))
    print("Done:", name)
```

Extract the ZIPs:

```python
import zipfile

for name in ["WIDER_train.zip", "WIDER_val.zip", "wider_face_split.zip"]:
    with zipfile.ZipFile(os.path.join(save_dir, name)) as z:
        z.extractall(save_dir)
print("Extraction complete")
```

### Step 3 — Convert Annotations to YOLO Format

```python
import cv2
import scipy.io as sio
from tqdm import tqdm

def convert_annotations(mat_file, img_root, label_root):
    annots        = sio.loadmat(mat_file)
    event_list    = annots["event_list"]
    file_list     = annots["file_list"]
    face_bbx_list = annots["face_bbx_list"]

    for i, event in enumerate(tqdm(event_list)):
        event_name = event[0][0]
        files  = file_list[i][0]
        bboxes = face_bbx_list[i][0]

        for j, file in enumerate(files):
            filename   = file[0][0]
            img_path   = os.path.join(img_root,   event_name, filename + ".jpg")
            label_file = os.path.join(label_root, filename + ".txt")

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            with open(label_file, "w") as f:
                for bbox in bboxes[j][0]:
                    x, y, bw, bh = bbox
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    bw /= w
                    bh /= h
                    f.write(f"0 {x_center} {y_center} {bw} {bh}\n")

convert_annotations(os.path.join(split_path, "wider_face_train.mat"), train_img_path, train_label_path)
convert_annotations(os.path.join(split_path, "wider_face_val.mat"),   val_img_path,   val_label_path)
print("Annotation conversion complete")
```

### Step 4 — Upload Dataset to S3

```python
import boto3
import os

region    = "cn-northwest-1"
bucket    = "YOLO"
s3_prefix = "wider_subset"
s3_client = boto3.client("s3", region_name=region)
local_base = r"C:\Users\your_user\YOLO\WIDER_subset"

# Upload data.yaml
s3_client.upload_file(
    os.path.join(local_base, "data.yaml"), bucket, f"{s3_prefix}/data.yaml"
)

def upload_dir(local_dir, bucket, s3_prefix, subfolder):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path   = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            s3_key     = f"{s3_prefix}/{subfolder}/{rel_path}"
            s3_client.upload_file(local_path, bucket, s3_key)

upload_dir(os.path.join(local_base, "WIDER_train", "images"), bucket, s3_prefix, "WIDER_train/images")
upload_dir(os.path.join(local_base, "WIDER_train", "labels"), bucket, s3_prefix, "WIDER_train/labels")
upload_dir(os.path.join(local_base, "WIDER_val",   "images"), bucket, s3_prefix, "WIDER_val/images")
upload_dir(os.path.join(local_base, "WIDER_val",   "labels"), bucket, s3_prefix, "WIDER_val/labels")
print("Dataset uploaded to S3")
```

### Step 5 — Package and Upload Training Script

```bash
tar -czf code.tar.gz train.py requirements.txt
aws s3 cp code.tar.gz s3://YOLO/wider_subset/code.tar.gz
```

### Step 6 — Launch SageMaker Training Job

```python
import sagemaker
from sagemaker.pytorch import PyTorch

region    = "cn-northwest-1"
role      = "arn:aws-cn:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE"
bucket    = "YOLO"
s3_prefix = "wider_subset"
sess      = sagemaker.Session()

estimator = PyTorch(
    entry_point       = "train.py",
    source_dir        = f"s3://{bucket}/{s3_prefix}/code.tar.gz",
    role              = role,
    framework_version = "2.0",
    py_version        = "py310",
    instance_type     = "ml.g5.xlarge",
    instance_count    = 1,
    output_path       = f"s3://{bucket}/{s3_prefix}/output/",
    hyperparameters   = {
        "epochs": 100,
        "imgsz" : 640,
        "model" : "/opt/ml/input/data/weights/yolov8n.pt",
        "data"  : "/opt/ml/input/data/training/data.yaml"
    }
)

estimator.fit({
    "training": f"s3://{bucket}/{s3_prefix}/",
    "weights" : f"s3://{bucket}/{s3_prefix}/yolov8n.pt"
})
```

---

## 📊 Results

### Performance Comparison across Epochs

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| 5     | 0.707     | 0.380  | 0.441   | 0.225         |
| **50**| **0.834** | **0.413** | **0.498** | **0.262** |
| 100   | 0.814     | 0.410  | 0.490   | 0.267         |

### Key Observations

- ✅ **50 Epochs = optimal** — highest Precision (0.834) and mAP@0.5 (0.498)
- ⚠️ **100 Epochs shows slight overfitting** — Precision drops slightly
- 📈 **Recall plateaus at ~0.41** — limited by dataset size and small face instances
- ⬆️ **mAP@0.5:0.95 marginally improves at 100 Epochs** (0.267 > 0.262) — good generalisation at stricter IoU thresholds

---

## 📚 References

1. Jocher G, Chaurasia A, Qiu J. *Ultralytics YOLOv8 Documentation.*
2. Redmon J, Farhadi A. *YOLO9000: Better, Faster, Stronger.* CVPR, 2017.
3. Yang S, Luo P, Loy C C, Tang X. *WIDER FACE: A Face Detection Benchmark.* CVPR, 2016.
4. Amazon Web Services. *Amazon SageMaker Developer Guide.*
5. Amazon Web Services. *Amazon S3 User Guide.*
6. Goodfellow I, Bengio Y, Courville A. *Deep Learning.* MIT Press, 2016.

---

<a name="chinese"></a>

---

# 🎯 基于AWS SageMaker与YOLOv8n的人脸检测系统

## 📖 项目概述

本项目基于 **YOLOv8n** 轻量级目标检测模型，在 **WIDER Face** 数据集子集上进行人脸检测训练，完整训练流程托管于 **AWS SageMaker**，数据通过 **Amazon S3** 统一管理。

最佳模型（50 Epoch）指标如下：

| 指标 | 数值 |
|------|------|
| Precision | **0.834** |
| Recall | 0.413 |
| mAP@0.5 | **0.498** |
| mAP@0.5:0.95 | 0.262 |

## 📁 目录结构

```
YOLO-Face-Detection/
├── src/
│   └── train.py                  # SageMaker训练入口脚本
├── docs/
│   ├── YOLO_code_ready_v2.docx   # 完整技术报告（中文）
│   └── YOLO_english.docx         # 完整技术报告（英文）
├── requirements.txt
└── README.md
```

## ⚙️ 环境配置

### 软件环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| Python | 3.10 |
| PyTorch | 2.0 |
| Ultralytics | 8.2.0 |
| OpenCV | 4.8 |
| SageMaker SDK | Latest |

### 硬件环境（AWS SageMaker）

| 项目 | 配置 |
|------|------|
| 云平台 | AWS SageMaker |
| 实例类型 | ml.g5.xlarge |
| GPU | NVIDIA A10G |
| GPU显存 | 24GB |
| CPU | 4 vCPU |
| 内存 | 16GB |

## 🚀 使用说明

详细步骤请参阅完整技术报告：
- 中文版：[`docs/YOLO_code_ready_v2.docx`](docs/YOLO_code_ready_v2.docx)
- 英文版：[`docs/YOLO_english.docx`](docs/YOLO_english.docx)

主要流程：
1. 安装依赖环境
2. 下载并解压 WIDER Face 数据集
3. 将标注格式转换为 YOLO 格式
4. 上传数据集至 S3
5. 打包训练脚本并上传 S3
6. 启动 SageMaker Training Job
7. 查看训练结果与指标分析

## 📊 实验结果

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| 5     | 0.707     | 0.380  | 0.441   | 0.225         |
| **50**| **0.834** | **0.413** | **0.498** | **0.262** |
| 100   | 0.814     | 0.410  | 0.490   | 0.267         |

## 📚 参考文献

1. Jocher G, Chaurasia A, Qiu J. *Ultralytics YOLOv8 Documentation.*
2. Redmon J, Farhadi A. *YOLO9000: Better, Faster, Stronger.* CVPR, 2017.
3. Yang S, et al. *WIDER FACE: A Face Detection Benchmark.* CVPR, 2016.
4. Amazon Web Services. *Amazon SageMaker Developer Guide.*
5. Amazon Web Services. *Amazon S3 User Guide.*

---

<p align="center">Made with ❤️ using YOLOv8n + AWS SageMaker</p>
