"""Unit tests for the Table-2 metrics, checked against hand-worked examples."""
import numpy as np
import torch

from src.eval import ac_at_k, avg_at_5, evaluate, mrr, ranks_of_truth


class TestRanks:
    def test_best_score_is_rank_one(self):
        assert ranks_of_truth([[0.9, 0.5, 0.1]], [0])[0] == 1

    def test_worst_score_is_last_rank(self):
        assert ranks_of_truth([[0.9, 0.5, 0.1]], [2])[0] == 3

    def test_ties_broken_pessimistically(self):
        # All equal: the truth could be 1st optimistically, but we report 3rd so
        # an untrained model cannot look good by accident.
        assert ranks_of_truth([[0.5, 0.5, 0.5]], [0])[0] == 3

    def test_accepts_torch_tensors(self):
        r = ranks_of_truth(torch.tensor([[0.1, 0.9]]), torch.tensor([1]))
        assert r[0] == 1


class TestMetrics:
    def test_perfect_ranking(self):
        scores = [[0.9, 0.5, 0.1], [0.2, 0.8, 0.1]]
        m = evaluate(scores, [0, 1])
        assert m.ac1 == 1.0 and m.mrr == 1.0 and m.avg5 == 1.0

    def test_worst_ranking(self):
        # Truth last of 3 -> rank 3. AC@1=0, AC@3=1, MRR=1/3.
        m = evaluate([[0.9, 0.5, 0.1]], [2])
        assert m.ac1 == 0.0 and m.ac3 == 1.0 and np.isclose(m.mrr, 1 / 3)

    def test_avg5_hand_computed(self):
        # rank 3 -> AC@1..5 = 0,0,1,1,1 -> mean = 3/5
        assert np.isclose(avg_at_5([[0.9, 0.5, 0.1]], [2]), 0.6)

    def test_ac_at_k_monotonic(self):
        rng = np.random.default_rng(0)
        s, t = rng.random((40, 12)), rng.integers(0, 12, 40)
        vals = [ac_at_k(s, t, k) for k in range(1, 6)]
        assert all(a <= b for a, b in zip(vals, vals[1:]))

    def test_mrr_between_zero_and_one(self):
        rng = np.random.default_rng(1)
        assert 0.0 < mrr(rng.random((50, 12)), rng.integers(0, 12, 50)) <= 1.0

    def test_half_correct(self):
        m = evaluate([[0.9, 0.1], [0.9, 0.1]], [0, 1])
        assert m.ac1 == 0.5 and np.isclose(m.mrr, (1 + 0.5) / 2)
