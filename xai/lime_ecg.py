"""
xai/lime_ecg.py
---------------
LIME (Local Interpretable Model-agnostic Explanations) adapted for
1D ECG time-series classification (Ribeiro et al., 2016).

Key design choices matching the paper:
  - Super-features: non-overlapping 50-sample windows (~167 ms @ 300 Hz)
  - 5,000 perturbed samples per explanation
  - Gaussian noise (from training distribution) for masking
  - Ridge regression surrogate with exponential kernel (σ=0.75)
  - Attribution = absolute ridge coefficient per super-feature
  - Five independent runs averaged to reduce sampling variance

Usage:
    explainer = LIMEExplainer(model, background_mean, background_std)
    attributions = explainer.explain(x, class_idx=1)  # (3000,)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from typing import Optional, Tuple, Union


class LIMEExplainer:
    """
    Parameters
    ----------
    model           : trained PyTorch model (eval mode expected)
    background_mean : per-timestep mean from training set  (3000,)
    background_std  : per-timestep std  from training set  (3000,)
    segment_len     : super-feature window size in samples (default 50)
    n_samples       : number of perturbation samples
    kernel_width    : exponential kernel bandwidth σ
    n_runs          : independent runs to average (reduces variance)
    alpha           : ridge regression regularisation strength
    device          : torch device
    """

    def __init__(
        self,
        model:           nn.Module,
        background_mean: np.ndarray,
        background_std:  np.ndarray,
        segment_len:     int   = 50,
        n_samples:       int   = 5000,
        kernel_width:    float = 0.75,
        n_runs:          int   = 5,
        alpha:           float = 1.0,
        device:          str   = "cpu",
    ) -> None:
        self.model      = model.eval()
        self.bg_mean    = background_mean
        self.bg_std     = background_std
        self.seg_len    = segment_len
        self.n_samples  = n_samples
        self.kw         = kernel_width
        self.n_runs     = n_runs
        self.alpha      = alpha
        self.device     = torch.device(device)

    def _n_segments(self, signal_len: int) -> int:
        return signal_len // self.seg_len

    def _perturb(
        self, x: np.ndarray, n_segments: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate *n_samples* binary masks and corresponding perturbed signals.

        Returns
        -------
        masks    : (n_samples, n_segments)  binary
        perturbs : (n_samples, signal_len)  perturbed signals
        """
        signal_len = len(x)
        masks      = np.random.randint(0, 2,
                                       size=(self.n_samples, n_segments),
                                       ).astype(np.float32)
        perturbs   = np.tile(x, (self.n_samples, 1)).astype(np.float32)

        for seg_idx in range(n_segments):
            off_mask   = masks[:, seg_idx] == 0          # (n_samples,)
            seg_start  = seg_idx * self.seg_len
            seg_end    = seg_start + self.seg_len

            if off_mask.any():
                noise = np.random.normal(
                    self.bg_mean[seg_start:seg_end],
                    self.bg_std[seg_start:seg_end] + 1e-8,
                    size=(off_mask.sum(), self.seg_len),
                )
                perturbs[off_mask, seg_start:seg_end] = noise

        return masks, perturbs

    def _predict(self, signals: np.ndarray) -> np.ndarray:
        """Run model on (N, T) array; return softmax probabilities (N, C)."""
        self.model.eval()
        results = []
        bs = 256
        # Use enable_grad() to ensure any internal hooks that expect gradients
        # (even if not used for attribution here) don't fail.
        with torch.enable_grad():
            for i in range(0, len(signals), bs):
                batch = torch.from_numpy(signals[i:i+bs]).unsqueeze(1)
                batch = batch.to(self.device)
                # Ensure input doesn't trigger retain_grad issues if hooks are present
                batch.requires_grad_(False)
                with torch.no_grad():
                    probs = torch.softmax(self.model(batch), dim=-1).cpu().numpy()
                results.append(probs)
        return np.concatenate(results, axis=0)

    def _kernel_weights(self, masks: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Exponential kernel: w = exp(-d² / σ²)."""
        distances = np.sqrt(np.sum((masks - original) ** 2, axis=1))
        return np.exp(-(distances ** 2) / (self.kw ** 2))

    def explain(
        self,
        x:           np.ndarray,
        class_idx:   int,
        return_time: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Compute LIME attribution for a single ECG segment.

        Parameters
        ----------
        x          : 1D array (signal_len,)
        class_idx  : class to explain
        return_time: if True return (attribution, wall_clock_seconds)

        Returns
        -------
        attribution : (signal_len,)  – super-feature values upsampled to signal
        """
        t0         = time.perf_counter()
        signal_len = len(x)
        n_segments = self._n_segments(signal_len)
        original   = np.ones((1, n_segments), dtype=np.float32)

        run_coefs = []
        for _ in range(self.n_runs):
            masks, perturbs = self._perturb(x, n_segments)
            probs           = self._predict(perturbs)         # (N, C)
            targets         = probs[:, class_idx]             # (N,)
            weights         = self._kernel_weights(masks, original)

            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(masks, targets, sample_weight=weights)
            run_coefs.append(np.abs(ridge.coef_))             # (n_segments,)

        coefs = np.mean(run_coefs, axis=0)                    # (n_segments,)

        # Upsample segment-level attributions to signal length
        attribution = np.repeat(coefs, self.seg_len)[:signal_len]

        elapsed = time.perf_counter() - t0
        return (attribution, elapsed) if return_time else attribution
