"""Hand-constructed scenarios where each loss's value is known."""
import math

import pytest
import torch

from src.losses import neighbour_mask, sco_loss, tcd_loss


def embed(direction):
    """A unit vector along the given axis, for exact cosine values."""
    v = torch.zeros(4)
    v[direction] = 1.0
    return v


class TestTCD:
    def _setup(self, root_diverged):
        """2 services; service 0 is the root cause."""
        h_norm = torch.stack([embed(0), embed(1)]).unsqueeze(0)
        if root_diverged:          # root orthogonal to its normal, other unchanged
            h_anom = torch.stack([embed(2), embed(1)]).unsqueeze(0)
        else:                      # root unchanged, other diverged
            h_anom = torch.stack([embed(0), embed(3)]).unsqueeze(0)
        return h_anom, h_norm, torch.tensor([0])

    def test_intuition_orientation_zero_when_root_diverges(self):
        h_anom, h_norm, y = self._setup(root_diverged=True)
        # cos_r=0, cos_other=1 -> relu(delta + 0 - 1) = 0 for delta < 1
        assert tcd_loss(h_anom, h_norm, y, delta=0.5).item() == 0.0

    def test_intuition_orientation_penalises_root_staying_normal(self):
        h_anom, h_norm, y = self._setup(root_diverged=False)
        # cos_r=1, cos_other=0 -> relu(0.5 + 1 - 0) = 1.5
        assert tcd_loss(h_anom, h_norm, y, delta=0.5).item() == pytest.approx(1.5)

    def test_as_written_is_the_reverse(self):
        h_anom, h_norm, y = self._setup(root_diverged=True)
        # literal Eq. 8: relu(0.5 - 0 + 1) = 1.5 for the scenario intuition rewards
        assert tcd_loss(h_anom, h_norm, y, delta=0.5, as_written=True).item() == pytest.approx(1.5)

    def test_root_term_excluded_from_sum(self):
        h = torch.stack([embed(0)]).unsqueeze(0)      # single service IS the root
        assert tcd_loss(h, h, torch.tensor([0]), delta=0.9).item() == 0.0


class TestNeighbourMask:
    def test_in_and_out_edges_counted_self_loops_ignored(self):
        #  0->1, 2->0, 0->0 (self), 3->4 (unrelated); root = 0
        ei = torch.tensor([[[0, 2, 0, 3], [1, 0, 0, 4]]])
        m = neighbour_mask(ei, torch.tensor([0]), n_services=5)
        assert m.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]

    def test_padding_ignored(self):
        ei = torch.tensor([[[0, -1], [1, -1]]])
        m = neighbour_mask(ei, torch.tensor([0]), n_services=3)
        assert m.tolist() == [[0.0, 1.0, 0.0]]


class TestSCO:
    def test_zero_when_root_beats_neighbours_by_margin(self):
        scores = torch.tensor([[0.9, 0.5, 0.7]])
        p = torch.tensor([[0.0, 1.0, 1.0]])
        assert sco_loss(scores, torch.tensor([0]), p, m=0.1).item() == 0.0

    def test_penalises_neighbour_within_margin(self):
        scores = torch.tensor([[0.9, 0.85, 0.1]])
        p = torch.tensor([[0.0, 1.0, 1.0]])
        # violation only from service 1: 0.1 - (0.9-0.85) = 0.05
        assert sco_loss(scores, torch.tensor([0]), p, m=0.1).item() == pytest.approx(0.05)

    def test_non_neighbours_ignored_even_if_higher(self):
        scores = torch.tensor([[0.5, 0.99]])
        p = torch.tensor([[0.0, 0.0]])        # service 1 not in P(r)
        assert sco_loss(scores, torch.tensor([0]), p, m=0.1).item() == 0.0


class TestGradients:
    def test_all_losses_backpropagate(self):
        """Would have caught the in-place write that broke autograd: every loss
        must survive backward() when its inputs require gradients."""
        torch.manual_seed(0)
        h_anom = torch.randn(2, 4, 8, requires_grad=True)
        h_norm = torch.randn(2, 4, 8, requires_grad=True)
        scores = torch.rand(2, 4, requires_grad=True)
        y = torch.tensor([0, 2])
        p = torch.ones(2, 4)
        loss = tcd_loss(h_anom, h_norm, y, delta=0.5) + sco_loss(scores, y, p, m=0.1)
        loss.backward()
        assert h_anom.grad is not None and scores.grad is not None
