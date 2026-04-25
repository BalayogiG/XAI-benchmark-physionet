"""
models/transformer.py
---------------------
Transformer encoder for ECG classification.

Architecture:
  Patch embedding  (stride-S 1D conv acting as a linear patch projection)
  → Sinusoidal positional encoding
  → N × TransformerEncoderLayer (multi-head self-attention + FFN)
  → CLS token pooling
  → Linear classifier

Hyperparameters match the paper:
  d_model=256, n_heads=8, n_layers=6, d_ff=1024
"""

import math
import torch
import torch.nn as nn
from typing import Optional


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal encoding (Vaswani et al., 2017).
    Supports arbitrary sequence length up to *max_len*.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── Patch embedding ───────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Project input patches of size *patch_size* to *d_model* via a strided
    1D convolution, equivalent to a linear projection of non-overlapping
    time patches.
    """

    def __init__(self, in_channels: int, d_model: int,
                 patch_size: int = 30) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, T/patch_size, d_model)"""
        return self.proj(x).transpose(1, 2)


# ── Transformer ───────────────────────────────────────────────────────────────

class ECGTransformer(nn.Module):
    """
    Parameters
    ----------
    in_channels : input leads
    num_classes : output classes
    d_model     : transformer hidden dimension
    n_heads     : number of attention heads
    n_layers    : encoder depth
    d_ff        : feedforward expansion dimension
    patch_size  : number of time samples per patch (3000/30 = 100 patches)
    dropout     : attention and FFN dropout rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        d_model: int     = 256,
        n_heads: int     = 8,
        n_layers: int    = 6,
        d_ff: int        = 1024,
        patch_size: int  = 30,
        dropout: float   = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)
        self.pos_enc     = SinusoidalPositionalEncoding(d_model, dropout)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,      # pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)
        Returns logits (B, num_classes)
        """
        B = x.size(0)

        # Patch embedding + positional encoding
        patches = self.patch_embed(x)                        # (B, S, d)
        patches = self.pos_enc(patches)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d)
        seq = torch.cat([cls, patches], dim=1)                # (B, S+1, d)

        # Transformer encoder
        enc = self.encoder(seq)                               # (B, S+1, d)

        # Classification from CLS token
        return self.head(enc[:, 0])


if __name__ == "__main__":
    model = ECGTransformer(in_channels=1, num_classes=4)
    x = torch.randn(8, 1, 3000)
    out = model(x)
    print("Transformer output shape:", out.shape)   # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
