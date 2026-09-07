"""Hybrid-Aware Graph Attention (H-GAT) -- the paper's actual novelty.

A standard GAT learns how much neighbour j matters to node i purely from the
node features. DynaCausal multiplies that learned attention by a second,
*observed* quantity: the Stage-1 dynamic edge weight e_ij, derived from how much
traffic actually flowed on the i<-j call in this window and how much of it
errored.

    attn_ij = softmax_{j in N(i)}( LeakyReLU( a^T [W h_i || W h_j] ) )
    h_i'    = sigmoid( sum_{j in N(i)} attn_ij * e_ij * W h_j )

So the model can *learn* "this neighbour is informative", but reality still gets
a vote: if no traffic crossed that edge this window, e_ij is small and the
message is damped no matter what attention wanted. That product is why this
cannot be expressed with torch_geometric's built-in GATConv, and why we
subclass MessagePassing directly.

Two separate quantities in the paper share the symbol alpha. Here they are
strictly `attn` (learned attention) and `edge_weight`/`e_ij` (observed, from
Stage 1). They never collide.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class HGATLayer(MessagePassing):
    """One H-GAT layer.

    Args:
        in_dim:      input feature width per node.
        out_dim:     output width *per head*.
        heads:       number of attention heads.
        concat:      True -> output heads*out_dim (intermediate layers),
                     False -> average the heads (final layer).
        negative_slope: LeakyReLU slope used in the attention logit.
        dropout:     dropout applied to attention coefficients.
        apply_sigmoid: the paper writes sigma(...) around the aggregation.
                     Kept configurable because a hard sigmoid on every layer
                     saturates in deep stacks; the paper's best K is 2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        apply_sigmoid: bool = True,
        bias: bool = True,
    ) -> None:
        # aggr="add" is the sum over j in N(i) in the equation above.
        # node_dim=0 because our node feature tensor is (N, heads, out_dim);
        # propagation must index the node axis, not the default last axis.
        super().__init__(aggr="add", node_dim=0)

        self.in_dim, self.out_dim = in_dim, out_dim
        self.heads, self.concat = heads, concat
        self.negative_slope, self.dropout = negative_slope, dropout
        self.apply_sigmoid = apply_sigmoid

        # W in the equations. One shared projection, reshaped per head.
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        # `a`, split into its source and destination halves so the logit can be
        # computed as a_src·Wh_j + a_dst·Wh_i without materialising the
        # concatenation for every edge.
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """
        Args:
            x:           (N, in_dim) node features.
            edge_index:  (2, E) COO edges. Row 0 = source j, row 1 = target i.
                         Direction matters: e_ij != e_ji.
            edge_weight: (E,) the Stage-1 dynamic weights e_ij, in [0, 1].
        Returns:
            (N, heads*out_dim) if concat else (N, out_dim).
        """
        if edge_weight.dim() != 1 or edge_weight.size(0) != edge_index.size(1):
            raise ValueError(
                f"edge_weight must be (E,) matching edge_index (2,E); "
                f"got {tuple(edge_weight.shape)} vs {tuple(edge_index.shape)}"
            )

        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_dim)      # (N, H, C)

        # Per-node halves of the attention logit; the edge-wise sum happens in
        # message(), where they arrive gathered as _j (source) and _i (target).
        alpha_src = (h * self.att_src).sum(-1)                 # (N, H)
        alpha_dst = (h * self.att_dst).sum(-1)                 # (N, H)

        out = self.propagate(
            edge_index,
            x=h,
            alpha=(alpha_src, alpha_dst),
            edge_weight=edge_weight,
            size=(n, n),
        )                                                       # (N, H, C)

        out = out.reshape(n, self.heads * self.out_dim) if self.concat else out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        # sigma(...) from the paper, applied after aggregation.
        return torch.sigmoid(out) if self.apply_sigmoid else out

    def message(
        self,
        x_j: Tensor,            # (E, H, C) source features W h_j
        alpha_j: Tensor,        # (E, H)    a_src · W h_j
        alpha_i: Tensor,        # (E, H)    a_dst · W h_i
        edge_weight: Tensor,    # (E,)      e_ij from Stage 1
        index: Tensor,          # (E,)      target node per edge, for the softmax
        ptr,
        dim_size,
    ) -> Tensor:
        # Standard GAT logit, then softmax normalised over each target's
        # in-neighbourhood N(i) -- `index` is what scopes the softmax per node.
        alpha = torch.nn.functional.leaky_relu(alpha_j + alpha_i, self.negative_slope)
        attn = softmax(alpha, index, ptr, dim_size)
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)

        # THE non-standard step: learned attention AND observed edge weight.
        # Note e_ij multiplies *after* the softmax, so rows no longer sum to 1 --
        # that is intended. An edge carrying no traffic should contribute little
        # in absolute terms, not merely little relative to its siblings.
        coeff = attn * edge_weight.unsqueeze(-1)               # (E, H)
        return x_j * coeff.unsqueeze(-1)                       # (E, H, C)


class HGAT(nn.Module):
    """Stack of K H-GAT layers. The paper reports performance peaking at K=2."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(
                HGATLayer(in_dim, out_dim, heads=heads, concat=False,
                          dropout=dropout, apply_sigmoid=apply_sigmoid)
            )
        else:
            self.layers.append(
                HGATLayer(in_dim, hidden_dim, heads=heads, concat=True,
                          dropout=dropout, apply_sigmoid=apply_sigmoid)
            )
            for _ in range(n_layers - 2):
                self.layers.append(
                    HGATLayer(hidden_dim * heads, hidden_dim, heads=heads, concat=True,
                              dropout=dropout, apply_sigmoid=apply_sigmoid)
                )
            # Final layer averages heads to land exactly on out_dim (D_spat).
            self.layers.append(
                HGATLayer(hidden_dim * heads, out_dim, heads=heads, concat=False,
                          dropout=dropout, apply_sigmoid=apply_sigmoid)
            )
        self.out_dim = out_dim

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        """(N, in_dim) -> (N, out_dim), one row per service."""
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x
