"""Scoring head: service embedding -> suspicion score in [0, 1].

    S = sigmoid( W2 · ReLU(W1 · H + b1) + b2 )

Design decision (paper ambiguity #3). Eq. 7 writes W1 · H with H in R^(N x D),
which does not say whether W1 is shared across services or applied to the
flattened matrix. We apply it **per node with shared weights**, i.e. row-wise.

Reasons: (a) it is consistent with the paper's notation read as a row-wise
linear map; (b) it makes the model topology-agnostic, so one implementation
runs D1's 12 services and D2's 50 without changing shapes; (c) a flattened
variant would hard-code N and make cross-dataset transfer impossible.
Recorded in README as a reimplementation decision.
"""
from __future__ import annotations

from torch import Tensor, nn


class ScoringHead(nn.Module):
    """Two-layer MLP applied independently to each service's embedding."""

    def __init__(self, d_in: int, d_hidden: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        # nn.Linear maps the LAST axis, so passing (..., d_in) already gives the
        # shared-across-services behaviour we want -- no reshaping needed.
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: Tensor, return_logits: bool = False) -> Tensor:
        """
        Args:
            h: (..., d_in) service embeddings.
            return_logits: if True, skip the sigmoid and return raw logits.
        Returns:
            (...) scores, one scalar per service.

        Note: cross-entropy over services needs *logits*, so training calls this
        with return_logits=True and lets the loss apply log_softmax. The sigmoid
        of Eq. 7 is used for the [0,1] scores that ranking and SCO consume.
        """
        out = self.net(h).squeeze(-1)
        return out if return_logits else out.sigmoid()
