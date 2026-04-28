"""
End-to-end XAI benchmark runner with structured logging.
"""

import os
import sys
import argparse
import json
import csv
import logging
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


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "benchmark.log")

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-run
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # File
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_CLASSES = {
    "cnn1d":       CNN1D,
    "resnet1d":    ResNet1D,
    "bilstm":      BiLSTM,
    "transformer": ECGTransformer,
}

XAI_APPLICABILITY = {
    "cnn1d":       ["lime", "kernel_shap", "deep_shap", "gradcam", "gradcam_pp"],
    "resnet1d":    ["lime", "kernel_shap", "deep_shap", "gradcam", "gradcam_pp"],
    "bilstm":      ["lime", "kernel_shap"],
    "transformer": ["lime", "kernel_shap", "deep_shap"],
}

TARGET_CLASS = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(name: str, ckpt_path: str, device: torch.device, logger):
    logger.debug(f"Loading model {name} from {ckpt_path}")
    model = MODEL_CLASSES[name](in_channels=1, num_classes=4)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    logger.info(f"Model {name} loaded successfully")
    return model.eval().to(device)


def load_test_data(data_dir: str, logger):
    path = os.path.join(data_dir, "test.npz")
    logger.debug(f"Loading test data from {path}")
    data = np.load(path)
    logger.info(f"Test data loaded: {data['X'].shape}")
    return data["X"], data["y"]


def load_background(data_dir: str, logger, n: int = 200, seed: int = 42):
    path = os.path.join(data_dir, "train.npz")
    logger.debug(f"Loading background data from {path}")
    data = np.load(path)

    X = data["X"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    bg = X[idx]

    logger.info(f"Background sampled: {bg.shape}")
    return bg, bg.mean(axis=0), bg.std(axis=0)


def build_explain_fns(
    model_name: str,
    model: torch.nn.Module,
    bg_X: np.ndarray,
    bg_mean: np.ndarray,
    bg_std: np.ndarray,
    device: torch.device,
    logger
) -> Dict[str, callable]:

    applicable = XAI_APPLICABILITY[model_name]
    explainers = {}

    logger.debug(f"Building explainers for {model_name}: {applicable}")

    if "lime" in applicable:
        lime = LIMEExplainer(model, bg_mean, bg_std, device=str(device))
        explainers["lime"] = lambda x, return_time=False, _l=lime: \
            _l.explain(x, TARGET_CLASS, return_time=return_time)

    if "kernel_shap" in applicable:
        kshap = KernelSHAPExplainer(model, bg_X, bg_mean, bg_std, device=str(device))
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

    logger.info(f"Explainers ready: {list(explainers.keys())}")
    return explainers


# ── Main Benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    model_dir: str,
    data_dir: str,
    output_dir: str,
    n_samples: int = 200,
    device_str: str = "auto",
) -> None:

    logger = setup_logger(output_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )

    logger.info(f"Benchmark device: {device}")

    X_test, y_test = load_test_data(data_dir, logger)
    bg_X, bg_mean, bg_std = load_background(data_dir, logger)

    rows: List[dict] = []

    for model_name in MODEL_CLASSES.keys():
        ckpt = os.path.join(model_dir, f"{model_name}_best.pt")

        if not os.path.exists(ckpt):
            logger.warning(f"Checkpoint not found: {ckpt}. Skipping.")
            continue

        logger.info("=" * 60)
        logger.info(f"Model: {model_name.upper()}")
        logger.info("=" * 60)

        try:
            model = load_model(model_name, ckpt, device, logger)
        except Exception:
            logger.exception(f"Failed to load model: {model_name}")
            continue

        explain_fns = build_explain_fns(
            model_name, model, bg_X, bg_mean, bg_std, device, logger
        )

        for xai_name, explain_fn in explain_fns.items():
            logger.info(f"[{xai_name}] evaluating on {n_samples} samples...")

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
                    row[f"{metric}_std"] = round(std, 4)
                    logger.info(f"{metric:12s}: {mean:.4f} ± {std:.4f}")

                rows.append(row)

            except Exception:
                logger.exception(f"{xai_name} failed on {model_name}")

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    fieldnames = [
        "model", "xai_method",
        "fidelity_mean", "fidelity_std",
        "sparsity_mean", "sparsity_std",
        "stability_mean", "stability_std",
        "runtime_mean", "runtime_std"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    logger.info(f"JSON saved: {json_path}")
    logger.info("Benchmark completed successfully")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../checkpoints")
    parser.add_argument("--data_dir", default="../processed_data")
    parser.add_argument("--output_dir", default="../results")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    run_benchmark(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        device_str=args.device,
    )