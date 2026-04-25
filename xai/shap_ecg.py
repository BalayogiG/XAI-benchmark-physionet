"""
xai/shap_ecg.py
---------------
SHAP explanations for ECG classification: KernelSHAP and DeepSHAP.

KernelSHAP (Lundberg & Lee, 2017)
  - Model-agnostic; estimates Shapley values via weighted least squares
  - Background set: 200 training samples
  - Attributions computed at 50-sample super-feature level

DeepSHAP (Shrikumar et al., 2017 + Lundberg & Lee, 2017)
  - Differentiable models only (CNN, ResNet, Transformer)
  - DeepLIFT + Shapley theory; implemented via shap.DeepExplainer
  - Faster than KernelSHAP; same background set

Usage:
    explainer = KernelSHAPExplainer(model, background_X)
    attrs = explainer.explain(x, class_idx=1)   # (3000,)

    explainer = DeepSHAPExplainer(model, background_X, device="cuda")
    attrs = explainer.explain(x, class_idx=1)   # (3000,)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import shap
from typing import Callable, Tuple, Union


# ── Helper: segment-level wrapper ────────────────────────────────────────────

def _make_segment_predictor(
    model:       nn.Module,
    x_original:  np.ndarray,
    bg_mean:     np.ndarray,
    bg_std:      np.ndarray,
    seg_len:     int,
    class_idx:   int,
    device:      torch.device,
) -> Callable:
    """
    Return a function that:
      1. Takes a (N, n_segments) binary presence array
      2. Reconstructs full signals (absent segments → Gaussian noise)
      3. Returns model probability for *class_idx*

    Used internally by KernelSHAP.
    """
    signal_len = len(x_original)
    n_segments = signal_len // seg_len

    def predict_fn(masks: np.ndarray) -> np.ndarray:
        n = len(masks)
        signals = np.tile(x_original, (n, 1)).astype(np.float32)
        for seg in range(n_segments):
            off = masks[:, seg] == 0
            if off.any():
                s, e = seg * seg_len, (seg + 1) * seg_len
                noise = np.random.normal(
                    bg_mean[s:e], bg_std[s:e] + 1e-8,
                    size=(off.sum(), seg_len)
                )
                signals[off, s:e] = noise

        results, bs = [], 256
        model.eval()
        # Use enable_grad context to safely handle any lingering hooks
        with torch.enable_grad():
            for i in range(0, n, bs):
                batch = torch.from_numpy(signals[i:i+bs]).unsqueeze(1).to(device)
                batch.requires_grad_(False)
                with torch.no_grad():
                    probs = torch.softmax(model(batch), -1).cpu().numpy()
                results.append(probs[:, class_idx])
        return np.concatenate(results)

    return predict_fn


# ── KernelSHAP ───────────────────────────────────────────────────────────────

# ── KernelSHAP ───────────────────────────────────────────────────────────────

class KernelSHAPExplainer:
    """
    Improved KernelSHAP for 1D ECG time-series explanations.
    
    Key improvements over original:
      - Larger but reasonable background (80 samples) to avoid degeneracy
      - l1_reg="num_features(30)" to prevent LassoLarsIC errors
      - Higher nsamples for better approximation
      - Balanced speed vs. stability for ECG signals (3000 timesteps)
    """

    def __init__(
        self,
        model:        nn.Module,
        background_X: np.ndarray,
        bg_mean:      np.ndarray,
        bg_std:       np.ndarray,
        seg_len:      int = 75,      # Increased from 50 → fewer super-features (more stable)
        device:       str = "cpu",
    ) -> None:
        self.model    = model.eval()
        self.bg_X     = background_X
        self.bg_mean  = bg_mean
        self.bg_std   = bg_std
        self.seg_len  = seg_len
        self.device   = torch.device(device)

    def explain(
        self,
        x:           np.ndarray,
        class_idx:   int,
        return_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        t0         = time.perf_counter()
        signal_len = len(x)
        n_segments = signal_len // self.seg_len

        predict_fn = _make_segment_predictor(
            self.model, x, self.bg_mean, self.bg_std,
            self.seg_len, class_idx, self.device
        )

        # ── Stable Background Creation ─────────────────────────────────────
        rng = np.random.default_rng(42)
        
        # Use 80 background samples — good balance between stability and speed
        # (avoids both "too few" degeneracy and "too many" speed warning)
        n_bg = 80
        bg_seg = rng.integers(0, 2, size=(n_bg, n_segments)).astype(np.float32)

        explainer = shap.KernelExplainer(predict_fn, bg_seg)

        x_seg = np.ones((1, n_segments), dtype=np.float32)

        # ── SHAP call with stability improvements ──────────────────────────
        shap_v = explainer.shap_values(
            x_seg,
            nsamples=512,                    # Good approximation quality
            l1_reg="num_features(30)",       # Prevents degenerate regressors
            silent=True
        )

        # Handle both single-output and multi-class cases
        if isinstance(shap_v, list):
            # Multi-class case: pick the desired class
            coefs = np.abs(np.array(shap_v[class_idx]).flatten())
        else:
            coefs = np.abs(np.array(shap_v).flatten())

        # Ensure correct length (in case of any resizing issues)
        if len(coefs) != n_segments:
            coefs = np.resize(coefs, n_segments)

        # Upsample segment-level attributions back to original signal length
        attribution = np.repeat(coefs, self.seg_len)[:signal_len]
        elapsed     = time.perf_counter() - t0

        return (attribution, elapsed) if return_time else attribution


# ── DeepSHAP (via Captum DeepLift) ───────────────────────────────────────────

def _replace_inplace_relu(model: nn.Module) -> nn.Module:
    """
    Captum's DeepLift requires that each ReLU module is used exactly once
    in the network (so it can attach unique hooks). ResNet reuses a single
    self.relu instance across multiple forward calls, which triggers:
      'A Module ReLU was detected that does not contain some of the
       input/output attributes required for DeepLift computations.'

    Fix: recursively replace every ReLU(inplace=True) with a fresh
    ReLU(inplace=False) instance so each is unique.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        else:
            _replace_inplace_relu(module)
    return model

def _fix_resnet_relu_reuse(model: nn.Module) -> nn.Module:
    """
    Specific fix for architectures like ResNet where a single self.relu
    is used multiple times in the forward pass.
    """
    # First, handle BasicBlocks in ResNet
    for stage_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, stage_name):
            stage = getattr(model, stage_name)
            for block in stage:
                if hasattr(block, 'relu'):
                    block.relu = nn.ReLU(inplace=False)
                if hasattr(block, 'relu1'):
                    block.relu1 = nn.ReLU(inplace=False)
                if hasattr(block, 'relu2'):
                    block.relu2 = nn.ReLU(inplace=False)
    
    # Also handle the main model's relu if it exists
    if hasattr(model, 'relu'):
        model.relu = nn.ReLU(inplace=False)
        
    return model


class DeepSHAPExplainer:
    """
    DeepLIFT-based attribution via Captum, approximating DeepSHAP by
    averaging attributions over a sample of background references.

    Parameters
    ----------
    model       : trained PyTorch model (eval mode)
    background_X: (n_bg, signal_len) training samples as baselines
    n_baselines : number of background samples to average over
    device      : torch device string
    """

    def __init__(
        self,
        model:        nn.Module,
        background_X: np.ndarray,
        n_baselines:  int = 50,
        device:       str = "cpu",
    ) -> None:
        import copy
        from captum.attr import DeepLift

        self.device = torch.device(device)

        # Deep-copy the model so we don't mutate the original,
        # then fix shared/inplace ReLU modules for Captum compatibility.
        model_copy = copy.deepcopy(model).eval()
        _fix_resnet_relu_reuse(model_copy)
        _replace_inplace_relu(model_copy)
        self.model = model_copy
        self.dl    = DeepLift(model_copy)

        rng = np.random.default_rng(0)
        idx = rng.choice(len(background_X),
                         size=min(n_baselines, len(background_X)),
                         replace=False)
        bg = background_X[idx].astype(np.float32)
        self.baselines = torch.from_numpy(bg).unsqueeze(1).to(self.device)

    def explain(
        self,
        x:           np.ndarray,
        class_idx:   int,
        return_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        import warnings
        t0  = time.perf_counter()
        x_t = torch.from_numpy(
            x[np.newaxis].astype(np.float32)
        ).unsqueeze(1).to(self.device)   # (1, 1, T)
        
        # Explicitly set requires_grad to suppress Captum's "Input Tensor 0 did not already require gradients" warning
        x_t.requires_grad_(True)

        attrs_list = []
        with torch.enable_grad():
            with warnings.catch_warnings():
                # Suppress Captum's informational warning about setting/removing hooks on non-linear activations
                warnings.filterwarnings("ignore", message=".*Setting forward, backward hooks and attributes.*")
                for i in range(self.baselines.shape[0]):
                    baseline = self.baselines[i:i+1]
                    attr = self.dl.attribute(
                        x_t,
                        baselines=baseline,
                        target=class_idx,
                    )
                    attrs_list.append(attr.detach().cpu().numpy()[0, 0])

        attribution = np.abs(np.mean(attrs_list, axis=0))
        elapsed     = time.perf_counter() - t0
        return (attribution, elapsed) if return_time else attribution
