"""
utils/seed.py
-------------
Global seed setter for full reproducibility across NumPy, PyTorch,
Python random, and CUDA operations.

Usage:
    from utils.seed import set_seed
    set_seed(42)
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Fix all random seeds for reproducibility.

    Parameters
    ----------
    seed               : integer seed value
    deterministic_cudnn: if True, forces CuDNN into deterministic mode
                         (may reduce GPU performance slightly)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def get_worker_init_fn(seed: int = 42):
    """
    Returns a worker_init_fn for DataLoader that seeds each worker
    independently but reproducibly.

    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(42))
    """
    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return _worker_init_fn
