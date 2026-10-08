"""The three training losses: CE + TCD + SCO (paper Sec. 3.4).

Label interface: every term takes the root-cause index `labels` and, for SCO,
a neighbour mask derived from the call graph. Phase 2's label-free variant
swaps in computed pseudo-labels through the same two arguments; nothing here
knows where they came from.

TCD sign note (documented finding): the paper's Eq. 8 as printed,
    max(0, delta - cos(H_r_anom, H_r_norm) + cos(H_i_anom, H_i_norm)),
is minimised by keeping the root cause SIMILAR to its own normal state, which
is the reverse of the intuition stated beside it ("magnifies the temporal
divergence ... of true root-cause services, while suppressing such divergence
for non-root ones"). Default here follows the intuition; `as_written=True`
implements the literal equation so both can be ablated.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def tcd_loss(h_anom: Tensor, h_norm: Tensor, labels: Tensor,
             delta: float = 0.2, as_written: bool = False) -> Tensor:
    """Temporal causal disentanglement, averaged over the batch.

    Args:
        h_anom: (B, N, D) embeddings of the anomaly window.
        h_norm: (B, N, D) embeddings of the same services' normal window.
        labels: (B,) root-cause index per case.
        delta:  margin.
        as_written: use the paper's literal Eq. 8 orientation.
    """
    cos = F.cosine_similarity(h_anom, h_norm, dim=-1)          # (B, N)
    b = torch.arange(cos.size(0), device=cos.device)
    cos_r = cos[b, labels].unsqueeze(1)                        # (B, 1)
    if as_written:
        viol = F.relu(delta - cos_r + cos)   # wants root similar, others far
    else:
        viol = F.relu(delta + cos_r - cos)   # wants root far, others similar
    # The i != r sum. Multiply by a mask rather than writing into `viol`
    # in place: an in-place write on relu's output breaks autograd.
    keep = torch.ones_like(viol)
    keep[b, labels] = 0.0
    return (viol * keep).sum(dim=1).mean()


def neighbour_mask(edge_index: Tensor, labels: Tensor, n_services: int) -> Tensor:
    """P(r) as the root cause's 1-hop neighbourhood in the window's call graph.

    Design decision: the paper never defines P(r). We use services with an
    observed call to or from the root cause in this window (self-loops
    excluded). Documented in README; the radius is the natural knob later.

    Args:
        edge_index: (B, 2, E_b) per-sample edges, NOT batch-offset, possibly
                    padded with -1 columns.
    Returns:
        (B, N) float mask, 1 where i is in P(r).
    """
    bsz = labels.size(0)
    mask = torch.zeros(bsz, n_services)
    for s in range(bsz):
        ei = edge_index[s]
        r = int(labels[s])
        for k in range(ei.size(1)):
            i, j = int(ei[0, k]), int(ei[1, k])
            if i < 0 or i == j:
                continue                      # padding or self-loop
            if i == r:
                mask[s, j] = 1.0
            elif j == r:
                mask[s, i] = 1.0
        mask[s, r] = 0.0
    return mask


def sco_loss(scores: Tensor, labels: Tensor, p_mask: Tensor, m: float = 0.1) -> Tensor:
    """Spatial causal ordering: S_r must beat every affected service by m.

    Args:
        scores: (B, N) sigmoid scores S.
        p_mask: (B, N) 1 where the service is in P(r).
    """
    b = torch.arange(scores.size(0), device=scores.device)
    s_r = scores[b, labels].unsqueeze(1)                       # (B, 1)
    viol = F.relu(m - (s_r - scores)) * p_mask
    return viol.sum(dim=1).mean()


def total_loss(logits: Tensor, scores: Tensor, h_anom: Tensor, h_norm: Tensor,
               labels: Tensor, p_mask: Tensor, lambda1: float, lambda2: float,
               delta: float, m: float, tcd_as_written: bool = False):
    """L = L_CE + lambda1 * L_TCD + lambda2 * L_SCO. Returns (total, parts)."""
    ce = F.cross_entropy(logits, labels)
    tcd = tcd_loss(h_anom, h_norm, labels, delta, tcd_as_written)
    sco = sco_loss(scores, labels, p_mask, m)
    return ce + lambda1 * tcd + lambda2 * sco, {"ce": ce.item(), "tcd": tcd.item(), "sco": sco.item()}
