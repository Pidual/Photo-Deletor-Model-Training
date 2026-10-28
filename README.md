<div align="center">

# 📸 Photo Deletor — Model Training

**A ResNet-50 image classifier that decides whether a photo is _memorable_ or _forgettable_,
so the junk in your camera roll can be cleaned up automatically.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-export-005CED?logo=onnx&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)

</div>

---

## ✨ Overview

This repo holds the training side of **Photo Deletor**. It fine-tunes an ImageNet-pretrained
**ResNet-50** on two classes:

| Class | Meaning | Examples |
|-------|---------|----------|
| 🗑️ `forgetable` | Safe to delete | random screenshots, memes, blurry shots, forwarded chat images |
| ⭐ `memorable`  | Worth keeping | personal photos, people, places, moments |

The trained model is exported to **ONNX** so the app can run it outside of PyTorch.

## 📊 Results

Trained for 20 epochs at **512×512** on ~69k images (80/20 train/val split):

| Metric | Value |
|--------|-------|
| Final training accuracy | **98.6 %** |
| Final training loss | 0.0385 |
| Confusion-matrix accuracy (13,839 images) | **98.8 %** |

```
                 Predicted
               Forgettable  Memorable
Actual Forget.     4190        101
       Memor.        59       9489
```

> [!NOTE]
> The confusion matrix was computed on a fresh random split made after training, so some of
> those images were also seen during training. Treat it as optimistic.

## 🗂️ Project Structure

```
Photo-Deletor-Model-Training/
├── notebooks/
│   ├── train_classifier.ipynb           # Full training run (20 epochs) + evaluation + visual tests
│   ├── train_classifier_fast_run.ipynb  # Same pipeline, 3 epochs, for quick experiments
│   └── resolution_search.ipynb          # Compares input resolutions (224 → 650) by val loss
├── scripts/
│   └── export_onnx.py                   # Converts the .pth weights to ONNX
├── models/
│   ├── resnet50_finetuned.pth           # V1 weights
│   ├── resnet50_finetunedV2.pth         # V2 weights (current)
│   └── resnet50_classifier.onnx         # ONNX export of V2
├── dataset/
│   ├── images_dataset/                  # Training data (not in git — see below)
│   └── visual_testing/                  # Handful of images for eyeballing predictions
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

> [!TIP]
> For GPU training, install the CUDA build of PyTorch from
> [pytorch.org](https://pytorch.org/get-started/locally/) before running the line above.

### 2. Prepare the dataset

The training images are not committed. Put them in `dataset/images_dataset/`, one folder per
class (the `torchvision.datasets.ImageFolder` layout):

```
dataset/images_dataset/
├── forgetable/
│   └── *.jpg / *.png
└── memorable/
    └── *.jpg / *.png
```

### 3. Train

Open `notebooks/train_classifier.ipynb` and run all cells. It will:

1. Load the dataset and split it 80/20
2. Load ResNet-50 with ImageNet weights, freeze the backbone and train only the final layer
3. Train with Adam, `ReduceLROnPlateau` and early stopping
4. Save the weights to `models/resnet50_finetunedV2.pth`
5. Plot a confusion matrix and show predictions for `dataset/visual_testing/`

Use `train_classifier_fast_run.ipynb` for a quick 3-epoch sanity check.

### 4. Export to ONNX

```bash
python scripts/export_onnx.py
```

This writes `models/resnet50_classifier.onnx` and checks that it is valid.

## 🧠 Model Details

| | |
|---|---|
| **Backbone** | ResNet-50 (`IMAGENET1K_V1`), frozen |
| **Head** | `Linear(2048 → 2)` |
| **Input** | `1×3×512×512`, RGB, ImageNet-normalized |
| **Augmentation** | Random horizontal flip, ±10° rotation |
| **Optimizer** | Adam, lr `1e-4`, weight decay `1e-5` |
| **Loss** | Cross-entropy |
| **ONNX** | opset 14, input `image`, output `prediction` (logits), dynamic batch size |

### Using the ONNX model

The image must be preprocessed the same way as during training:

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("models/resnet50_classifier.onnx")

img = Image.open("photo.jpg").convert("RGB").resize((512, 512))
x = np.asarray(img, dtype=np.float32) / 255.0
x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
x = x.transpose(2, 0, 1)[None].astype(np.float32)   # HWC -> 1xCxHxW

logits = session.run(None, {"image": x})[0][0]
probs = np.exp(logits) / np.exp(logits).sum()
label = ["forgetable", "memorable"][probs.argmax()]
print(f"{label} ({probs.max():.1%})")
```
