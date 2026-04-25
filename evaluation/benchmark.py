"""
evaluation/benchmark.py
-----------------------
End-to-end XAI benchmark runner.

Loads all trained model checkpoints, instantiates each applicable
XAI method, evaluates all metrics, and writes results to CSV.

Usage:
    python evaluation/benchmark.py \
        --model_dir  ../checkpoints \
        --data_dir   ../processed_data \
        --output_dir ../results \
        --n_samples  200
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Ensure sibling dirs are importable when running from evaluation/
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT / "models"), str(ROOT / "xai"), str(ROOT / "evaluation")])

from cnn1d       import CNN1D
from resnet1d    import ResNet1D
from bilstm      import BiLSTM
from transformer import ECGTransformer

from lime_ecg    import LIMEExplainer
from shap_ecg    import KernelSHAPExplainer, DeepSHAPExplainer
from gradcam_ecg import GradCAM, GradCAMPlusPlus
from metrics     import evaluate_xai_method


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "cnn1d":       CNN1D,
    "resnet1d":    ResNet1D,
    "bilstm":      BiLSTM,
    "transformer": ECGTransformer,
}

# Which XAI methods are applicable per architecture
XAI_APPLICABILITY = {
    "cnn1d":       ["lime", "kernel_shap", "deep_shap", "gradcam", "gradcam_pp"],
    "resnet1d":    ["lime", "kernel_shap", "deep_shap", "gradcam", "gradcam_pp"],
    "bilstm":      ["lime", "kernel_shap"],
    "transformer": ["lime", "kernel_shap", "deep_shap"],
}

TARGET_CLASS = 1   # Atrial Fibrillation (class index 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(name: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = MODEL_CLASSES[name](in_channels=1, num_classes=4)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_test_data(data_dir: str):
    data = np.load(os.path.join(data_dir, "test.npz"))
    return data["X"], data["y"]


def load_background(data_dir: str, n: int = 200, seed: int = 42):
    """Sample *n* records from training set as SHAP/LIME background."""
    data = np.load(os.path.join(data_dir, "train.npz"))
    X    = data["X"]
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(X), size=min(n, len(X)), replace=False)
    bg   = X[idx]
    return bg, bg.mean(axis=0), bg.std(axis=0)


def build_explain_fns(
    model_name: str,
    model:      torch.nn.Module,
    bg_X:       np.ndarray,
    bg_mean:    np.ndarray,
    bg_std:     np.ndarray,
    device:     torch.device,
) -> Dict[str, callable]:
    """Instantiate all applicable XAI explainers for *model_name*."""
    applicable = XAI_APPLICABILITY[model_name]
    explainers = {}

    if "lime" in applicable:
        lime = LIMEExplainer(model, bg_mean, bg_std,
                             device=str(device))
        explainers["lime"] = lambda x, return_time=False, _l=lime: \
            _l.explain(x, TARGET_CLASS, return_time=return_time)

    if "kernel_shap" in applicable:
        kshap = KernelSHAPExplainer(model, bg_X, bg_mean, bg_std,
                                    device=str(device))
        explainers["kernel_shap"] = lambda x, return_time=False, _k=kshap: \
            _k.explain(x, TARGET_CLASS, return_time=return_time)

    if "deep_shap" in applicable:
        dshap = DeepSHAPExplainer(model, bg_X, device=str(device))
        explainers["deep_shap"] = lambda x, return_time=False, _d=dshap: \
            _d.explain(x, TARGET_CLASS, return_time=return_time)

    if "gradcam" in applicable:
        cam = GradCAM(model, model.get_cam_target_layer(), device=str(device))
        explainers["gradcam"] = lambda x, return_time=False, _c=cam: \
            _c.explain(x, TARGET_CLASS, return_time=return_time)

    if "gradcam_pp" in applicable:
        cam_pp = GradCAMPlusPlus(model, model.get_cam_target_layer(), device=str(device))
        explainers["gradcam_pp"] = lambda x, return_time=False, _c=cam_pp: \
            _c.explain(x, TARGET_CLASS, return_time=return_time)

    return explainers


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    model_dir:  str,
    data_dir:   str,
    output_dir: str,
    n_samples:  int = 200,
    device_str: str = "auto",
) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"[INFO] Benchmark device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    X_test, y_test = load_test_data(data_dir)
    bg_X, bg_mean, bg_std = load_background(data_dir)

    rows: List[dict] = []

    for model_name in MODEL_CLASSES.keys():
        ckpt = os.path.join(model_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt):
            print(f"[WARN] Checkpoint not found: {ckpt}. Skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*60}")

        model    = load_model(model_name, ckpt, device)
        explain_fns = build_explain_fns(
            model_name, model, bg_X, bg_mean, bg_std, device
        )

        for xai_name, explain_fn in explain_fns.items():
            print(f"  [{xai_name}] evaluating on {n_samples} samples ...")
            try:
                result = evaluate_xai_method(
                    explain_fn=explain_fn,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    class_idx=TARGET_CLASS,
                    device=str(device),
                    n_samples=n_samples,
                )

                row = {"model": model_name, "xai_method": xai_name}
                for metric, (mean, std) in result.items():
                    row[f"{metric}_mean"] = round(mean, 4)
                    row[f"{metric}_std"]  = round(std, 4)
                    print(f"    {metric:12s}: {mean:.4f} ± {std:.4f}")

                rows.append(row)

            except Exception as exc:
                print(f"  [ERROR] {xai_name} on {model_name}: {exc}")

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = ["model", "xai_method",
                  "fidelity_mean", "fidelity_std",
                  "sparsity_mean", "sparsity_std",
                  "stability_mean", "stability_std",
                  "runtime_mean", "runtime_std"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Save JSON too
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n[INFO] Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  default="../checkpoints")
    parser.add_argument("--data_dir",   default="../processed_data")
    parser.add_argument("--output_dir", default="../results")
    parser.add_argument("--n_samples",  type=int, default=200)
    parser.add_argument("--device",     default="auto")
    args = parser.parse_args()

    run_benchmark(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device_str=args.device,
    )
