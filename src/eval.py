"""Evaluation metrics for root cause ranking (paper Table 2).

Every metric takes per-case score vectors and the index of the true root cause,
ranks services by score descending, and asks where the truth landed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import Tensor


def _as_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def ranks_of_truth(scores, targets) -> np.ndarray:
    """1-based rank of the true root cause in each case.

    Args:
        scores:  (B, N) suspicion scores.
        targets: (B,) index of the true root-cause service.
    Returns:
        (B,) ranks, where 1 means the model put the truth first.

    Ties are broken pessimistically: a service tied with the truth is counted as
    ranking above it. Optimistic tie-breaking would inflate AC@1 on an untrained
    model whose scores are near-identical, so this is the honest choice.
    """
    scores, targets = _as_numpy(scores), _as_numpy(targets).astype(int)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (B, N), got {scores.shape}")
    if targets.shape[0] != scores.shape[0]:
        raise ValueError("scores and targets disagree on batch size")

    truth = scores[np.arange(scores.shape[0]), targets]
    # Count strictly-better plus tied-but-not-self, then convert to a 1-based rank.
    better = (scores > truth[:, None]).sum(axis=1)
    tied = (scores == truth[:, None]).sum(axis=1) - 1
    return better + tied + 1


def ac_at_k(scores, targets, k: int) -> float:
    """AC@k -- fraction of cases where the truth is in the top k."""
    return float((ranks_of_truth(scores, targets) <= k).mean())


def avg_at_5(scores, targets) -> float:
    """Avg@5 = (1/5) * sum_{k=1..5} AC@k.

    This is the paper's own definition and is unusual -- it averages the
    *cumulative* accuracies, not per-rank precision, so it is dominated by the
    easy high-k terms. Implemented exactly as written so our numbers are
    comparable to Table 2.
    """
    r = ranks_of_truth(scores, targets)
    return float(np.mean([(r <= k).mean() for k in range(1, 6)]))


def mrr(scores, targets) -> float:
    """Mean reciprocal rank."""
    return float((1.0 / ranks_of_truth(scores, targets)).mean())


@dataclass
class RCAMetrics:
    ac1: float
    ac3: float
    ac5: float
    avg5: float
    mrr: float
    n_cases: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AC@1={self.ac1:.3f}  AC@3={self.ac3:.3f}  AC@5={self.ac5:.3f}  "
            f"Avg@5={self.avg5:.3f}  MRR={self.mrr:.3f}  (n={self.n_cases})"
        )


def evaluate(scores, targets) -> RCAMetrics:
    """Compute the full Table-2 metric row for a set of cases."""
    r = ranks_of_truth(scores, targets)
    return RCAMetrics(
        ac1=float((r <= 1).mean()),
        ac3=float((r <= 3).mean()),
        ac5=float((r <= 5).mean()),
        avg5=float(np.mean([(r <= k).mean() for k in range(1, 6)])),
        mrr=float((1.0 / r).mean()),
        n_cases=int(r.shape[0]),
    )


# Paper Table 2, for regression-testing our reproduction.
PAPER_TARGETS = {
    "d1": RCAMetrics(ac1=0.769, ac3=0.980, ac5=1.000, avg5=0.937, mrr=0.873, n_cases=30),
    "d2": RCAMetrics(ac1=0.481, ac3=0.696, ac5=0.819, avg5=0.680, mrr=0.626, n_cases=285),
}
