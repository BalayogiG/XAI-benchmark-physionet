# Benchmarking XAI Methods for Physiological Signal Classification
## LIME vs SHAP vs Grad-CAM on ECG Arrhythmia Detection

> Code repository accompanying the paper:  
> **"Benchmarking Explainable AI Methods for Physiological Signal Classification:
> A Comparative Study of LIME, SHAP, and Grad-CAM on ECG Arrhythmia Detection"**

---

## Project Structure

```
xai_ecg_benchmark/
├── data/
│   ├── download.py          # PhysioNet 2017 dataset download script
│   └── preprocessing.py     # Filtering, segmentation, SMOTE, splits
├── models/
│   ├── cnn1d.py             # 1D-CNN architecture
│   ├── resnet1d.py          # ResNet-18 (1D adaptation)
│   ├── bilstm.py            # Bidirectional LSTM with attention
│   ├── transformer.py       # Transformer encoder
│   └── train.py             # Unified training loop
├── xai/
│   ├── lime_ecg.py          # LIME for time-series ECG
│   ├── shap_ecg.py          # KernelSHAP + DeepSHAP
│   └── gradcam_ecg.py       # Grad-CAM + Grad-CAM++
├── evaluation/
│   ├── metrics.py           # Fidelity, sparsity, stability, runtime
│   ├── clinical_alignment.py# IoU vs cardiologist annotations
│   └── benchmark.py         # Full benchmark runner
├── utils/
│   ├── visualization.py     # Saliency overlay plots
│   └── seed.py              # Reproducibility helpers
├── notebooks/
│   └── full_pipeline.ipynb  # End-to-end walkthrough
├── results/                 # Generated outputs (figures, CSVs)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/xai-ecg-benchmark.git
cd xai-ecg-benchmark
pip install -r requirements.txt
```

---

## Quick Start

### 1. Download & Preprocess Data
```bash
python data/download.py --output_dir ./raw_data
python data/preprocessing.py --input_dir ./raw_data --output_dir ./processed_data
```

### 2. Train All Models
```bash
python models/train.py --model all --data_dir ./processed_data --epochs 50
```

### 3. Run Full XAI Benchmark
```bash
python evaluation/benchmark.py --model_dir ./checkpoints --data_dir ./processed_data
```

### 4. Generate Paper Figures
```bash
python utils/visualization.py --results_dir ./results
```

---

## Dataset

We use the **PhysioNet/CinC Challenge 2017** dataset (Clifford et al., 2017):
- 14,749 single-lead ECG recordings
- 4 classes: Normal (N), Atrial Fibrillation (AF), Other (O), Noisy (~)
- Sampled at 300 Hz; durations 9–61 s

Download at: https://physionet.org/content/challenge-2017/1.0.0/

