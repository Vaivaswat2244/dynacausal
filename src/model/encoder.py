"""Temporal encoder: per-service Transformer over the window's time steps.

Stage 2 of DynaCausal. Each service is encoded *independently* -- there is no
cross-service mixing here. That is deliberate: all spatial reasoning is the
H-GAT's job (see hgat.py), so this module only answers "what has this one
service been doing over the last T steps?".

Input  (B, N, T, D_in)   multi-modal features from Stage 1
Output (B, N, D_temp)    one embedding per service
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal position signal.

    Self-attention is permutation-invariant, so without this the encoder could
    not tell "CPU rose then latency rose" from the reverse. For root cause
    analysis that ordering is the entire signal, so positions are not optional.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # Geometric series of wavelengths; the log/exp form is the numerically
        # stable way to write 1 / 10000^(2i/d).
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Buffer, not parameter: fixed signal, but must follow .to(device).
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """x: (B', T, d_model)"""
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    """Transformer encoder applied independently to each service's time series.

    Args:
        d_in:      Stage-1 feature width (D_metrics + D_logs + D_traces).
        d_temp:    embedding width. Paper's best setting is 32.
        n_layers:  Transformer blocks. Paper's best setting is 2 (deeper overfits).
        n_heads:   attention heads.
        dropout:   dropout inside the Transformer blocks.
        pooling:   how the T step outputs collapse to one vector.
                   "mean" averages over time, "last" takes the final step.
                   NOTE: the paper does not specify this. Mean is the default
                   because it is less sensitive to a single noisy final step;
                   recorded as a reimplementation decision.
    """

    def __init__(
        self,
        d_in: int,
        d_temp: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        pooling: str = "mean",
        max_len: int = 512,
    ) -> None:
        super().__init__()
        if d_temp % n_heads != 0:
            raise ValueError(f"d_temp={d_temp} must be divisible by n_heads={n_heads}")
        if pooling not in ("mean", "last"):
            raise ValueError(f"unknown pooling: {pooling!r}")

        self.d_in, self.d_temp, self.pooling = d_in, d_temp, pooling

        # Raw feature width differs per dataset, so project into the model width
        # first. This is what keeps the same model usable on D1 and D2.
        self.input_proj = nn.Linear(d_in, d_temp)
        self.pos = PositionalEncoding(d_temp, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_temp,
            nhead=n_heads,
            dim_feedforward=dim_feedforward or 4 * d_temp,
            dropout=dropout,
            batch_first=True,   # (B, T, D) rather than (T, B, D)
            norm_first=True,    # pre-LN: markedly more stable at low depth
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, T, d_in) -- batch, services, time steps, features.
        Returns:
            (B, N, d_temp) -- one embedding per service.
        """
        if x.dim() != 4:
            raise ValueError(f"expected (B, N, T, d_in), got {tuple(x.shape)}")
        b, n, t, d = x.shape
        if d != self.d_in:
            raise ValueError(f"expected d_in={self.d_in}, got {d}")

        # Fold services into the batch axis so each service is encoded on its
        # own: attention then spans time only, never services.
        h = x.reshape(b * n, t, d)
        h = self.pos(self.input_proj(h))
        h = self.transformer(h)                      # (B*N, T, d_temp)

        h = h.mean(dim=1) if self.pooling == "mean" else h[:, -1, :]
        return h.reshape(b, n, self.d_temp)
