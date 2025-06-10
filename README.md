# 🌱 Plant Classifier: ResNet50 → Grad‑CAM Pipeline

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MuhammadAli2603/plant-classifier/blob/main/ImageClassification%26InterpretabilityPipeline.ipynb)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9-orange.svg)]() [![Captum](https://img.shields.io/badge/Captum-%3E%3D0.4-red.svg)]() [![Albumentations](https://img.shields.io/badge/Albumentations-%3E%3D1.0-yellow.svg)]()

---

## 🚀 Project Overview

An **end‑to‑end** image classification and interpretability demo built **entirely in Google Colab**, using:

- **Backbone**: ResNet50 fine‑tuned on the [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) (~54K images, 38 classes).  
- **Augmentations**: Albumentations for robust training transforms.  
- **Interpretability**: Grad‑CAM overlays via Captum show _where_ the model “looks.”  
- **Zero‑Setup Demo**: Run everything—data download, training, visualization, and artifact saving—in one notebook. No local GPU or Python install needed!

---

## 📂 Repository Structure
```text
plant-classifier/
├── ImageClassification&InterpretabilityPipeline.ipynb   ← Colab notebook (all steps)
├── src/
│   ├── data_loader.py        ← PyTorch Dataset + Albumentations
│   ├── train.py              ← Training & fine‑tuning scripts
│   └── interpret.py          ← Grad‑CAM visualization utilities
├── requirements.txt          ← Pinned dependencies for local runs
├── .gitignore                ← Excludes data, weights, caches
└── LICENSE                   ← MIT License

`````
Quickstart
1️⃣ Run in Colab (Recommended)
1. Click the Open in Colab badge or visit:
   https://colab.research.google.com/github/MuhammadAli2603/plant-classifier/blob/main/ImageClassification%26InterpretabilityPipeline.ipynb
2. In the Colab menu, choose Runtime → Run all.
3. Follow the notebook—each section is self‑contained:
   - Data download & stratified split
   - Dataset & augmentation definition
   - Baseline training (3 epochs)
   - Fine‑tuning deeper layers (3–5 more epochs)
   - Grad‑CAM visualizations
   - Save weights & metrics to Google Drive

No files to manage locally!

2️⃣ Local Setup (Optional)
If you’d prefer to run scripts on your machine:

```
git clone https://github.com/MuhammadAli2603/plant-classifier.git
cd plant-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can then open the notebook in Jupyter Lab or extract cells into standalone scripts.
Usage Highlights
- **DataLoader**
  ```python
  from src.data_loader import PlantDataset
  ds = PlantDataset(root_dir="path/to/PlantVillage-Dataset/raw/color")
  ```
- **Training**
  ```bash
  python src/train.py
  ```
- **Grad‑CAM**
  ```python
  from src.interpret import show_cam
  show_cam("path/to/sample.jpg", class_idx=5)
  ```
Sample Results
| Model                  | Val Accuracy |
|------------------------|--------------|
| ResNet50 (baseline)    | 0.89         |
| ResNet50 (fine‑tuned)  | 0.94         |

Critical Considerations
- Class Imbalance: Some classes have far fewer samples—consider weighted or focal loss.
- Compute Efficiency: ResNet50 is heavy; switching to EfficientNet‑B0 can cut runtime in half.
- Interpretation Validity: Always sanity‑check Grad‑CAM on blank or random inputs.

## 🤝 Contributing

1. Fork this repo
2. Make your changes (in Colab or locally)
3. Commit & push to your fork
4. Open a Pull Request describing your enhancements
License
This project is licensed under the MIT License. See LICENSE for details.


