"""Full DynaCausal model: temporal encoder -> H-GAT -> scoring head."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import TemporalEncoder
from .head import ScoringHead
from .hgat import HGAT


def batch_graphs(
    edge_indices: list[Tensor], edge_weights: list[Tensor], n_nodes: int
) -> tuple[Tensor, Tensor]:
    """Block-diagonalise per-sample graphs into one disconnected graph.

    Each sample in a batch has its own dynamic call graph (edges change per
    window). PyG handles this by offsetting sample b's node ids by b*N, so the
    samples become disjoint components of a single graph and no message ever
    crosses between them.

    Args:
        edge_indices: B tensors of shape (2, E_b), node ids in [0, n_nodes).
        edge_weights: B tensors of shape (E_b,).
        n_nodes:      services per sample (N).
    Returns:
        (2, sum(E_b)) offset edge_index and (sum(E_b),) edge_weight.
    """
    if len(edge_indices) != len(edge_weights):
        raise ValueError("edge_indices and edge_weights must have the same length")
    shifted = [ei + b * n_nodes for b, ei in enumerate(edge_indices)]
    return torch.cat(shifted, dim=1), torch.cat(edge_weights, dim=0)


class DynaCausal(nn.Module):
    """Ranks services by how likely each is the root cause of the incident.

    Pipeline, for one window:
      1. TemporalEncoder  (B, N, T, D_in) -> (B, N, D_temp)   per-service history
      2. HGAT             propagates across the weighted call graph -> (B, N, D_spat)
      3. ScoringHead      (B, N, D_spat) -> (B, N)            one score per service
    """

    def __init__(
        self,
        d_in: int,
        d_temp: int = 32,
        d_spat: int = 32,
        d_head: int = 32,
        n_enc_layers: int = 2,
        n_gat_layers: int = 2,
        enc_heads: int = 4,
        gat_heads: int = 4,
        gat_hidden: int = 16,
        dropout: float = 0.1,
        pooling: str = "mean",
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(
            d_in=d_in, d_temp=d_temp, n_layers=n_enc_layers,
            n_heads=enc_heads, dropout=dropout, pooling=pooling, max_len=max_len,
        )
        self.hgat = HGAT(
            in_dim=d_temp, hidden_dim=gat_hidden, out_dim=d_spat,
            n_layers=n_gat_layers, heads=gat_heads, dropout=dropout,
        )
        self.head = ScoringHead(d_in=d_spat, d_hidden=d_head, dropout=dropout)
        self.d_temp, self.d_spat = d_temp, d_spat

    def encode(self, x: Tensor) -> Tensor:
        """(B, N, T, D_in) -> (B, N, D_temp). Exposed because the TCD loss needs
        temporal embeddings of *normal* windows without running the graph."""
        return self.encoder(x)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        return_logits: bool = False,
        return_embeddings: bool = False,
    ):
        """
        Args:
            x:           (B, N, T, D_in)
            edge_index:  (2, E) node ids in [0, B*N), already batch-offset via
                         batch_graphs().
            edge_weight: (E,) Stage-1 dynamic edge weights.
        Returns:
            scores (B, N), plus (h_temp, h_spat) if return_embeddings.
        """
        b, n = x.shape[0], x.shape[1]

        h_temp = self.encoder(x)                       # (B, N, D_temp)
        # Flatten to node list for message passing, then restore the batch axis.
        h_spat = self.hgat(h_temp.reshape(b * n, -1), edge_index, edge_weight)
        h_spat = h_spat.reshape(b, n, self.d_spat)     # (B, N, D_spat)
        scores = self.head(h_spat, return_logits=return_logits)   # (B, N)

        if return_embeddings:
            return scores, h_temp, h_spat
        return scores
