"""Dynamic call graph construction (Stage 1).

For every window t we build a weighted *directed* graph over services:

    e_ij = sigmoid( edge_alpha * Norm(C_ij) + (1 - edge_alpha) * Norm(R_ij) )

  C_ij : request count on the i -> j call during the window
  R_ij : error rate (5xx / timeouts) on that call during the window
  Norm : min-max to [0, 1] across the edges present in the window

Direction matters: e_ij != e_ji.

The paper's ablation reports this as the single most important component, so the
arithmetic here is unit-tested in tests/test_graph.py.

Everything in this module operates on ONE window and holds no dataset state, so
the same code serves offline preprocessing and the Phase-2 online path that must
build a graph from a live telemetry window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def min_max_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max to [0, 1].

    Degenerate case: when every edge carries the same value the range is zero
    and the normalisation is undefined. We return zeros rather than 0.5 or NaN,
    so a window in which all edges are identical contributes no *differential*
    signal -- which is the honest reading of "nothing here distinguishes edges".
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return v
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < eps:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def compute_edge_weights(
    counts: np.ndarray,
    error_rates: np.ndarray,
    edge_alpha: float = 0.5,
    rescale: bool = False,
) -> np.ndarray:
    """The e_ij formula. Pure arithmetic, no graph structure -- easy to test.

    Args:
        counts:      (E,) request count C_ij per edge, raw (un-normalised).
        error_rates: (E,) error rate R_ij per edge, already a rate in [0, 1].
        edge_alpha:  blend between traffic volume and error rate. NOT the GAT
                     attention coefficient -- the paper overloads the symbol
                     alpha for both; here they never share a name.
        rescale:     see note below. Default False = faithful to the paper.

    Returns:
        (E,) edge weights.

    NOTE on the sigmoid's range. Both normalised terms lie in [0, 1], so the
    blend does too, and sigmoid maps [0, 1] -> [0.500, 0.731]. Taken literally
    the paper's edge weights therefore occupy a narrow band well away from zero:
    a completely idle edge still passes ~0.5 of its message, so e_ij modulates
    messages rather than gating them. We keep that behaviour by default for
    fidelity, and expose `rescale=True` to stretch the band back to [0, 1] as an
    ablation to run later.
    """
    if not 0.0 <= edge_alpha <= 1.0:
        raise ValueError(f"edge_alpha must be in [0,1], got {edge_alpha}")
    counts = np.asarray(counts, dtype=np.float64)
    error_rates = np.asarray(error_rates, dtype=np.float64)
    if counts.shape != error_rates.shape:
        raise ValueError(
            f"counts {counts.shape} and error_rates {error_rates.shape} must match"
        )
    if counts.size == 0:
        return np.empty(0, dtype=np.float64)

    blend = edge_alpha * min_max_norm(counts) + (1.0 - edge_alpha) * min_max_norm(error_rates)
    w = 1.0 / (1.0 + np.exp(-blend))
    if rescale:
        lo, hi = 0.5, 1.0 / (1.0 + np.exp(-1.0))   # sigmoid(0), sigmoid(1)
        w = (w - lo) / (hi - lo)
    return w


@dataclass
class CallGraph:
    """One window's directed, weighted call graph."""
    edge_index: Tensor      # (2, E) long; row 0 = source i, row 1 = target j
    edge_weight: Tensor     # (E,) float
    services: list[str]     # index -> service name

    def __len__(self) -> int:
        return int(self.edge_index.size(1))


def build_call_graph(
    calls: dict[tuple[str, str], tuple[float, float]],
    services: list[str],
    edge_alpha: float = 0.5,
    rescale: bool = False,
    add_self_loops: bool = True,
) -> CallGraph:
    """Build one window's graph from observed calls.

    Args:
        calls:    {(caller, callee): (request_count, error_count)} for the window.
        services: fixed service ordering; defines node ids. Passing the full
                  roster (not just services seen this window) keeps node ids
                  stable across windows, which the model relies on.
        edge_alpha: blend coefficient.
        add_self_loops: give each service an edge to itself with weight 1.0.
                  Without one, a service with no in-edges receives no message and
                  the H-GAT's softmax over an empty neighbourhood is undefined.

    Returns:
        CallGraph with edges oriented source -> target.
    """
    idx = {s: i for i, s in enumerate(services)}
    src, dst, cnts, errs = [], [], [], []

    for (caller, callee), (n_req, n_err) in calls.items():
        if caller not in idx or callee not in idx:
            continue          # ignore traffic to/from services outside the roster
        if n_req <= 0:
            continue          # no traffic observed -> no edge this window
        src.append(idx[caller])
        dst.append(idx[callee])
        cnts.append(float(n_req))
        errs.append(float(n_err) / float(n_req))    # R_ij is a *rate*

    weights = compute_edge_weights(
        np.array(cnts), np.array(errs), edge_alpha=edge_alpha, rescale=rescale
    )

    if add_self_loops:
        n = len(services)
        src.extend(range(n))
        dst.extend(range(n))
        weights = np.concatenate([weights, np.ones(n, dtype=np.float64)])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return CallGraph(
        edge_index=edge_index,
        edge_weight=torch.tensor(weights, dtype=torch.float32),
        services=list(services),
    )
