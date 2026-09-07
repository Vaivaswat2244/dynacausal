"""Shape and behaviour tests for the model, including H-GAT's defining property."""
import pytest
import torch

from src.model.dynacausal import DynaCausal, batch_graphs
from src.model.encoder import TemporalEncoder
from src.model.head import ScoringHead
from src.model.hgat import HGAT, HGATLayer


def ring(n):
    """A directed ring so every node has exactly one in-edge and one out-edge."""
    return torch.tensor([[i for i in range(n)], [(i + 1) % n for i in range(n)]])


class TestEncoder:
    def test_output_shape(self):
        out = TemporalEncoder(d_in=7, d_temp=32)(torch.randn(2, 5, 11, 7))
        assert out.shape == (2, 5, 32)

    def test_services_encoded_independently(self):
        """Changing service 0's data must not change service 1's embedding.

        This is what 'per service, independent' means: no information may leak
        across services in the temporal stage. All cross-service mixing is the
        H-GAT's job.
        """
        enc = TemporalEncoder(d_in=4, d_temp=8, dropout=0.0).eval()
        x = torch.randn(1, 3, 6, 4)
        before = enc(x)[0, 1]
        x2 = x.clone()
        x2[0, 0] = torch.randn(6, 4)          # scribble over service 0 only
        assert torch.allclose(before, enc(x2)[0, 1], atol=1e-6)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError):
            TemporalEncoder(d_in=4)(torch.randn(2, 5, 4))

    def test_rejects_indivisible_heads(self):
        with pytest.raises(ValueError):
            TemporalEncoder(d_in=4, d_temp=10, n_heads=4)


class TestHGAT:
    def test_output_shape(self):
        out = HGAT(16, 8, 12, n_layers=2, heads=2)(torch.randn(6, 16), ring(6), torch.rand(6))
        assert out.shape == (6, 12)

    def test_single_layer(self):
        out = HGAT(16, 8, 12, n_layers=1)(torch.randn(6, 16), ring(6), torch.rand(6))
        assert out.shape == (6, 12)

    def test_edge_weight_changes_output(self):
        """THE defining test: e_ij must actually affect the result.

        If this passes with a vanilla GATConv it would fail, because GATConv has
        no way to consume an observed per-edge weight. This is the single test
        that proves the custom MessagePassing subclass is doing its job.
        """
        layer = HGATLayer(8, 8, heads=2).eval()
        x, ei = torch.randn(5, 8), ring(5)
        out_a = layer(x, ei, torch.full((5,), 0.1))
        out_b = layer(x, ei, torch.full((5,), 0.9))
        assert not torch.allclose(out_a, out_b, atol=1e-5)

    def test_zero_edge_weight_silences_neighbours(self):
        """With all e_ij = 0 every incoming message is zeroed, so all nodes get
        the same output (bias only) regardless of their features."""
        layer = HGATLayer(8, 8, heads=1, apply_sigmoid=False, bias=False).eval()
        out = layer(torch.randn(5, 8), ring(5), torch.zeros(5))
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_direction_matters(self):
        """Reversing the edges changes the result: this is a directed graph."""
        layer = HGATLayer(8, 8, heads=1).eval()
        x, ei, ew = torch.randn(5, 8), ring(5), torch.rand(5)
        assert not torch.allclose(layer(x, ei, ew), layer(x, ei.flip(0), ew), atol=1e-5)

    def test_rejects_mismatched_edge_weight(self):
        with pytest.raises(ValueError):
            HGATLayer(8, 8)(torch.randn(5, 8), ring(5), torch.rand(3))


class TestHead:
    def test_shared_across_services(self):
        """Per-node shared weights: identical embeddings must score identically,
        wherever they sit. This is what makes the model topology-agnostic and
        lets one implementation run both D1 (12 services) and D2 (50)."""
        head = ScoringHead(d_in=6).eval()
        h = torch.randn(1, 6).repeat(4, 1)
        s = head(h)
        assert torch.allclose(s, s[0].expand_as(s), atol=1e-6)

    def test_accepts_any_service_count(self):
        head = ScoringHead(d_in=6).eval()
        assert head(torch.randn(2, 12, 6)).shape == (2, 12)
        assert head(torch.randn(2, 50, 6)).shape == (2, 50)

    def test_scores_in_unit_interval(self):
        s = ScoringHead(d_in=6).eval()(torch.randn(3, 9, 6) * 50)
        assert float(s.min()) >= 0.0 and float(s.max()) <= 1.0


class TestDynaCausal:
    def _batch(self, b, n, t, d):
        x = torch.randn(b, n, t, d)
        ei, ew = batch_graphs([ring(n)] * b, [torch.rand(n)] * b, n)
        return x, ei, ew

    def test_end_to_end_shape(self):
        m = DynaCausal(d_in=9)
        x, ei, ew = self._batch(3, 12, 20, 9)
        assert m(x, ei, ew).shape == (3, 12)

    def test_runs_on_both_dataset_sizes(self):
        """Same weights, different service counts -- D1 has 12, D2 has 50."""
        m = DynaCausal(d_in=9).eval()
        for n in (12, 50):
            x, ei, ew = self._batch(2, n, 15, 9)
            assert m(x, ei, ew).shape == (2, n)

    def test_gradients_reach_every_parameter(self):
        m = DynaCausal(d_in=9)
        x, ei, ew = self._batch(2, 12, 15, 9)
        torch.nn.functional.cross_entropy(
            m(x, ei, ew, return_logits=True), torch.tensor([0, 5])
        ).backward()
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_batch_graphs_keeps_samples_disjoint(self):
        """Sample b's edges must only touch sample b's nodes."""
        n, b = 6, 3
        ei, _ = batch_graphs([ring(n)] * b, [torch.rand(n)] * b, n)
        for s in range(b):
            block = ei[:, s * n:(s + 1) * n]
            assert block.min() >= s * n and block.max() < (s + 1) * n

    def test_batch_independence(self):
        """One sample's features must not affect another's scores."""
        m = DynaCausal(d_in=5, dropout=0.0).eval()
        x, ei, ew = self._batch(2, 8, 10, 5)
        before = m(x, ei, ew)[1]
        x2 = x.clone(); x2[0] = torch.randn(8, 10, 5)
        assert torch.allclose(before, m(x2, ei, ew)[1], atol=1e-5)
