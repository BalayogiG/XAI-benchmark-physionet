"""
xai/gradcam_ecg.py
------------------
Grad-CAM (Selvaraju et al., 2017) and Grad-CAM++ (Chattopadhay et al., 2018)
adapted for 1D ECG signals.

Both methods require access to feature maps from a convolutional layer.
Applicable to: CNN1D, ResNet1D.
NOT applicable to: BiLSTM, Transformer (use LIME/SHAP instead).

Grad-CAM:
  α_k = (1/Z) Σ_t ∂y_c/∂A^k_t      (global average of gradients)
  L   = ReLU( Σ_k α_k * A^k )       (weighted sum of feature maps)
  → upsample to input length via linear interpolation

Grad-CAM++:
  α_k = Σ_t [ (∂²y_c/∂(A^k_t)²) / (2∂²y_c/∂(A^k_t)² + Σ_t' A^k_t' ∂³y_c/∂(A^k_t)³) ]
           × ReLU(∂y_c/∂A^k_t)
  (simplified pixel-weight formulation; see Chattopadhay et al. eq. 19)

Usage:
    cam = GradCAM(model, target_layer=model.get_cam_target_layer())
    saliency = cam.explain(x, class_idx=1)    # (3000,)

    cam_pp = GradCAMPlusPlus(model, target_layer=model.get_cam_target_layer())
    saliency = cam_pp.explain(x, class_idx=1) # (3000,)
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class _BaseCAM:
    """
    Registers a forward hook on *target_layer* to capture feature maps
    and their gradients via retain_grad().

    The entire forward+backward pass is wrapped in torch.enable_grad()
    so it works correctly even when called inside a torch.no_grad() block.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 device: str = "cpu") -> None:
        self.model        = model.eval()
        self.target_layer = target_layer
        self.device       = torch.device(device)

        self._feature_maps: Optional[torch.Tensor] = None
        self._fmap_tensor:  Optional[torch.Tensor] = None

        def _save_features(module, input, output):
            # Capture feature maps
            self._fmap_tensor  = output
            self._feature_maps = output.detach().clone()

        def _save_grads(module, grad_in, grad_out):
            # Capture gradients
            self._grads = grad_out[0].detach().clone()

        self.model.eval()
        target_layer.register_forward_hook(_save_features)
        target_layer.register_full_backward_hook(_save_grads)

    def _run(self, x: torch.Tensor, class_idx: int) -> None:
        """Forward+backward under enable_grad so retain_grad() works."""
        with torch.enable_grad():
            self.model.zero_grad()
            # Ensure input requires grad for hooks to trigger correctly
            x.requires_grad_(True)
            output = self.model(x)
            score  = output[0, class_idx]
            score.backward()

    def _get_grads(self) -> torch.Tensor:
        if not hasattr(self, "_grads") or self._grads is None:
            raise RuntimeError("No gradients found. Call _run() first.")
        return self._grads   # (1, K, T')

    def _upsample(self, cam: np.ndarray, target_len: int) -> np.ndarray:
        t  = torch.from_numpy(cam[np.newaxis, np.newaxis].copy()).float()
        up = F.interpolate(t, size=target_len, mode="linear",
                           align_corners=False)
        return up.squeeze().numpy()


class GradCAM(_BaseCAM):
    """Standard Grad-CAM for 1D signals."""

    def explain(
        self,
        x:           np.ndarray,
        class_idx:   int,
        return_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        t0         = time.perf_counter()
        signal_len = len(x)

        x_t = torch.from_numpy(
            x[np.newaxis, np.newaxis].copy()
        ).float().to(self.device)
        x_t.requires_grad = True

        self._run(x_t, class_idx)

        grads = self._get_grads()[0]            # (K, T')
        fmaps = self._feature_maps[0]           # (K, T')

        weights = grads.mean(dim=-1)            # (K,)
        cam     = torch.relu(
            (weights[:, None] * fmaps).sum(dim=0)
        ).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        saliency = self._upsample(cam, signal_len)
        elapsed  = time.perf_counter() - t0
        return (saliency, elapsed) if return_time else saliency


class GradCAMPlusPlus(_BaseCAM):
    """Grad-CAM++ for 1D signals."""

    def explain(
        self,
        x:           np.ndarray,
        class_idx:   int,
        return_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        t0         = time.perf_counter()
        signal_len = len(x)

        x_t = torch.from_numpy(
            x[np.newaxis, np.newaxis].copy()
        ).float().to(self.device)
        x_t.requires_grad = True

        self._run(x_t, class_idx)

        grads = self._get_grads()[0]            # (K, T')
        fmaps = self._feature_maps[0]           # (K, T')

        grads_sq = grads ** 2
        grads_cu = grads ** 3
        denom    = (2.0 * grads_sq
                    + (fmaps * grads_cu).sum(dim=-1, keepdim=True))
        denom    = torch.where(denom.abs() > 1e-8, denom,
                               torch.ones_like(denom))
        alpha    = grads_sq / denom
        weights  = (alpha * torch.relu(grads)).sum(dim=-1)

        cam = torch.relu(
            (weights[:, None] * fmaps).sum(dim=0)
        ).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        saliency = self._upsample(cam, signal_len)
        elapsed  = time.perf_counter() - t0
        return (saliency, elapsed) if return_time else saliency
