"""
models/bilstm.py
----------------
Bidirectional LSTM with temporal attention for ECG classification.

Architecture:
  Input (B, 1, T) → reshape to (B, T, 1)
  → BiLSTM layer 1  (hidden_size per direction)
  → BiLSTM layer 2
  → Temporal Attention (soft-max weighted sum over time)
  → Linear classifier

The attention weights can serve as a lightweight baseline
explanation mechanism, though they are distinct from SHAP/LIME
attributions and are analysed separately in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class TemporalAttention(nn.Module):
    """
    Additive attention over the time dimension.

    Given hidden states H ∈ R^(B, T, D):
      e_t = tanh(W_h * h_t + b)
      α   = softmax(e)
      c   = sum_t α_t * h_t
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : (B, T, D)

        Returns
        -------
        context : (B, D)   – weighted sum
        weights : (B, T)   – attention weights (for visualisation)
        """
        energy  = torch.tanh(self.W(H))     # (B, T, D)
        scores  = self.v(energy).squeeze(-1) # (B, T)
        weights = F.softmax(scores, dim=-1)  # (B, T)
        context = (weights.unsqueeze(-1) * H).sum(dim=1)  # (B, D)
        return context, weights


class BiLSTM(nn.Module):
    """
    Parameters
    ----------
    in_channels  : input leads (1 for single-lead ECG)
    num_classes  : output classes
    hidden_size  : LSTM hidden units per direction
    num_layers   : number of stacked BiLSTM layers
    dropout      : dropout between LSTM layers
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        hidden_size: int = 128,
        num_layers: int  = 2,
        dropout: float   = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = TemporalAttention(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        x               : (B, C, T)  – standard conv-style input
        return_attention: if True, also return attention weights

        Returns
        -------
        logits  : (B, num_classes)
        weights : (B, T)  – only if return_attention=True
        """
        # Reshape: (B, C, T) → (B, T, C)
        x = x.permute(0, 2, 1)

        H, _ = self.lstm(x)           # (B, T, 2*H)
        context, weights = self.attention(H)
        logits = self.classifier(context)

        if return_attention:
            return logits, weights
        return logits


if __name__ == "__main__":
    model = BiLSTM(in_channels=1, num_classes=4, hidden_size=128)
    x = torch.randn(8, 1, 3000)
    out = model(x)
    print("BiLSTM output shape:", out.shape)   # (8, 4)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}")
