"""
models/train.py
---------------
Unified training script for all four architectures.

Usage:
    # Train a single model
    python models/train.py --model resnet1d --data_dir ./processed_data

    # Train all models sequentially
    python models/train.py --model all --data_dir ./processed_data --epochs 50
"""

import os
import argparse
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score

from cnn1d       import CNN1D
from resnet1d    import ResNet1D
from bilstm      import BiLSTM
from transformer import ECGTransformer


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "cnn1d":       CNN1D,
    "resnet1d":    ResNet1D,
    "bilstm":      BiLSTM,
    "transformer": ECGTransformer,
}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_split(data_dir: str, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(os.path.join(data_dir, f"{split}.npz"))
    X = torch.from_numpy(data["X"]).unsqueeze(1)   # (N, 1, T)
    y = torch.from_numpy(data["y"]).long()
    return X, y


def make_loader(X: torch.Tensor, y: torch.Tensor,
                batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y_batch.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    return {
        "loss":     total_loss / len(loader.dataset),
        "accuracy": (all_preds == all_labels).mean(),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "auc":      roc_auc_score(all_labels, all_probs,
                                  multi_class="ovr", average="macro"),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    data_dir:   str,
    output_dir: str,
    epochs:     int   = 50,
    batch_size: int   = 64,
    lr:         float = 3e-4,
    weight_decay: float = 1e-4,
    patience:   int   = 10,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}  |  Device: {device}")
    print(f"{'='*60}")

    # Data
    X_train, y_train = load_split(data_dir, "train")
    X_val,   y_val   = load_split(data_dir, "val")
    X_test,  y_test  = load_split(data_dir, "test")

    train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  batch_size, shuffle=False)

    # Model
    ModelClass = MODEL_REGISTRY[model_name]
    model      = ModelClass(in_channels=1, num_classes=4).to(device)
    total_p    = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_p:,}")

    # Class weights for imbalanced training
    class_counts = np.bincount(y_train.numpy())
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(device)
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    best_ckpt = os.path.join(output_dir, f"{model_name}_best.pt")
    best_f1   = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0       = time.time()
        tr_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        improved = val_metrics["f1_macro"] > best_f1
        if improved:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_ckpt)
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"tr_loss={tr_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"{'✓' if improved else ' '} | "
            f"{time.time()-t0:.1f}s"
        )

        if no_improve >= patience:
            print(f"[INFO] Early stopping at epoch {epoch}.")
            break

    # Test evaluation with best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n[Test] {model_name}: "
          f"acc={test_metrics['accuracy']:.4f}  "
          f"f1={test_metrics['f1_macro']:.4f}  "
          f"auc={test_metrics['auc']:.4f}")

    return {model_name: test_metrics}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--data_dir",    default="../processed_data")
    parser.add_argument("--output_dir",  default="../checkpoints")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=3e-4)
    args = parser.parse_args()

    models_to_train = (
        list(MODEL_REGISTRY.keys()) if args.model == "all"
        else [args.model]
    )

    all_results = {}
    for m in models_to_train:
        results = train_model(
            model_name=m,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        all_results.update(results)

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    for name, metrics in all_results.items():
        print(f"{name:12s}  acc={metrics['accuracy']:.4f}  "
              f"f1={metrics['f1_macro']:.4f}  auc={metrics['auc']:.4f}")
