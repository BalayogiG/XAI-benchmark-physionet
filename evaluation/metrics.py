"""
evaluation/metrics.py
---------------------
Quantitative evaluation metrics for XAI attributions.

Metrics implemented:
  1. Fidelity     – comprehensiveness: drop in P(y|x) when top-k features removed
  2. Sparsity     – Gini coefficient of absolute attribution magnitudes
  3. Stability    – mean Spearman ρ across perturbed neighbours
  4. Runtime      – wall-clock seconds per sample (measured externally, passed in)

All metrics return float scalars (higher = better for 1–3).
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from typing import Callable, Optional


# ── 1. Fidelity (Comprehensiveness) ──────────────────────────────────────────

def fidelity(
    model:        nn.Module,
    x:            np.ndarray,
    attribution:  np.ndarray,
    class_idx:    int,
    k:            int     = 600,   # top-k samples to remove  (≈ 20% of 3000)
    fill_value:   float   = 0.0,   # replacement value (mean = 0 after z-score)
    device:       str     = "cpu",
) -> float:
    """
    Comprehensiveness = P(y|x) − P(y|x_masked)

    The *k* timesteps with the highest |attribution| are replaced by
    *fill_value* (dataset mean after normalisation = 0).

    Returns
    -------
    float in [0, 1]; higher is better (more faithful attribution).
    """
    model.eval()
    dev = torch.device(device)

    def _predict(sig: np.ndarray) -> float:
        t = torch.from_numpy(sig[np.newaxis, np.newaxis]).float().to(dev)
        with torch.no_grad():
            prob = torch.softmax(model(t), -1)[0, class_idx].item()
        return prob

    p_original = _predict(x)

    top_k_idx  = np.argsort(np.abs(attribution))[-k:]
    x_masked   = x.copy()
    x_masked[top_k_idx] = fill_value

    p_masked   = _predict(x_masked)
    return float(p_original - p_masked)


# ── 2. Sparsity (Gini Coefficient) ───────────────────────────────────────────

def sparsity(attribution: np.ndarray) -> float:
    """
    Normalised Gini coefficient of |attribution|.

    G = 0  → perfectly uniform  (least sparse)
    G = 1  → single non-zero  (most sparse)

    Returns
    -------
    float in [0, 1]; higher is better.
    """
    vals = np.abs(attribution).flatten()
    if vals.sum() == 0:
        return 0.0
    vals_sorted = np.sort(vals)
    n    = len(vals_sorted)
    idx  = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * vals_sorted) / (n * vals_sorted.sum())) - (n + 1) / n
    return float(np.clip(gini, 0, 1))


# ── 3. Stability (Spearman ρ across neighbours) ───────────────────────────────

def stability(
    explain_fn:  Callable[[np.ndarray], np.ndarray],
    x:           np.ndarray,
    attribution: np.ndarray,
    n_neighbors: int   = 5,
    noise_std:   float = 0.01,
    seed:        int   = 0,
) -> float:
    """
    Mean Spearman rank-order correlation between the explanation of *x*
    and the explanations of *n_neighbors* Gaussian-perturbed versions.

    Parameters
    ----------
    explain_fn  : callable (x: np.ndarray) -> attribution np.ndarray
    x           : original 1D ECG segment  (T,)
    attribution : explanation of x         (T,)
    n_neighbors : number of perturbed neighbours
    noise_std   : Gaussian noise standard deviation

    Returns
    -------
    float in [-1, 1]; higher is better.
    """
    rng  = np.random.default_rng(seed)
    rhos = []
    # If the original attribution is constant, correlation is undefined.
    # We define stability as 0.0 in this case as the explanation is non-informative.
    if np.all(attribution == attribution[0]):
        return 0.0

    for _ in range(n_neighbors):
        x_noisy    = x + rng.normal(0, noise_std, size=x.shape).astype(np.float32)
        attr_noisy = explain_fn(x_noisy)
        
        # Check if the noisy attribution is constant
        if np.all(attr_noisy == attr_noisy[0]):
            rho = 0.0
        else:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(attribution, attr_noisy)
            
        rhos.append(rho if not np.isnan(rho) else 0.0)
    return float(np.mean(rhos))


# ── Batch evaluation helper ───────────────────────────────────────────────────

def evaluate_xai_method(
    explain_fn:  Callable,
    model:       nn.Module,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    class_idx:   int,
    device:      str = "cpu",
    n_samples:   int = 200,
    k_fidelity:  int = 600,
    seed:        int = 42,
) -> dict:
    """
    Run all four metrics on *n_samples* randomly drawn test examples.

    Parameters
    ----------
    explain_fn : callable (x: np.ndarray) -> (attribution, elapsed)
    model      : trained PyTorch model
    X_test     : (N, signal_len)
    y_test     : (N,)
    class_idx  : which class to explain
    n_samples  : how many test samples to evaluate

    Returns
    -------
    dict with keys: fidelity, sparsity, stability, runtime
                    each containing (mean, std).
    """
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(X_test), size=min(n_samples, len(X_test)),
                         replace=False)

    fids, spars, stabs, times = [], [], [], []

    for i in idx:
        x = X_test[i]

        attr, elapsed = explain_fn(x, return_time=True)

        fid  = fidelity(model, x, attr, class_idx, k=k_fidelity, device=device)
        spar = sparsity(attr)
        stab = stability(
            lambda xn: explain_fn(xn, return_time=False),
            x, attr
        )

        fids.append(fid)
        spars.append(spar)
        stabs.append(stab)
        times.append(elapsed)

    return {
        "fidelity":  (np.mean(fids),  np.std(fids)),
        "sparsity":  (np.mean(spars), np.std(spars)),
        "stability": (np.mean(stabs), np.std(stabs)),
        "runtime":   (np.mean(times), np.std(times)),
    }
