"""Unit tests for the dynamic call graph -- the paper's most important component."""
import numpy as np
import pytest
import torch

from src.data.graph import build_call_graph, compute_edge_weights, min_max_norm


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TestMinMaxNorm:
    def test_maps_to_unit_range(self):
        assert np.allclose(min_max_norm(np.array([10.0, 20.0, 30.0])), [0.0, 0.5, 1.0])

    def test_constant_input_returns_zeros(self):
        # Degenerate range: no edge is distinguishable, so no differential signal.
        assert np.allclose(min_max_norm(np.array([7.0, 7.0, 7.0])), [0.0, 0.0, 0.0])

    def test_empty(self):
        assert min_max_norm(np.array([])).size == 0


class TestEdgeWeights:
    def test_matches_formula_by_hand(self):
        counts = np.array([0.0, 50.0, 100.0])       # -> norm [0, 0.5, 1]
        errors = np.array([0.0, 0.0, 0.0])          # -> norm [0, 0, 0] (constant)
        got = compute_edge_weights(counts, errors, edge_alpha=1.0)
        assert np.allclose(got, sigmoid(np.array([0.0, 0.5, 1.0])))

    def test_alpha_zero_uses_only_error_rate(self):
        counts = np.array([1000.0, 1.0])            # ignored at alpha=0
        errors = np.array([0.0, 1.0])
        got = compute_edge_weights(counts, errors, edge_alpha=0.0)
        assert np.allclose(got, sigmoid(np.array([0.0, 1.0])))

    def test_alpha_one_uses_only_counts(self):
        counts = np.array([0.0, 10.0])
        errors = np.array([1.0, 0.0])               # ignored at alpha=1
        got = compute_edge_weights(counts, errors, edge_alpha=1.0)
        assert np.allclose(got, sigmoid(np.array([0.0, 1.0])))

    def test_blend_is_convex_combination(self):
        counts, errors = np.array([0.0, 100.0]), np.array([1.0, 0.0])
        a = 0.3
        # counts norm = [0,1], errors norm = [1,0]
        expect = sigmoid(np.array([a * 0 + (1 - a) * 1, a * 1 + (1 - a) * 0]))
        assert np.allclose(compute_edge_weights(counts, errors, edge_alpha=a), expect)

    def test_output_band_is_narrow_by_construction(self):
        # Documents a real property of the paper's formula: inputs are in [0,1],
        # so sigmoid confines every weight to [0.5, 0.731]. No edge is ever gated
        # off. If this test ever fails, the formula changed.
        w = compute_edge_weights(np.random.rand(50) * 1e4, np.random.rand(50), 0.5)
        assert w.min() >= 0.5 - 1e-9 and w.max() <= sigmoid(1.0) + 1e-9

    def test_rescale_stretches_to_unit_range(self):
        w = compute_edge_weights(np.array([0.0, 100.0]), np.array([0.0, 0.0]),
                                 edge_alpha=1.0, rescale=True)
        assert np.isclose(w.min(), 0.0) and np.isclose(w.max(), 1.0)

    def test_rejects_alpha_out_of_range(self):
        with pytest.raises(ValueError):
            compute_edge_weights(np.array([1.0]), np.array([0.0]), edge_alpha=1.5)

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            compute_edge_weights(np.array([1.0, 2.0]), np.array([0.0]))


class TestBuildCallGraph:
    def test_direction_is_preserved(self):
        # Only a -> b exists; b -> a must NOT appear. e_ij != e_ji.
        g = build_call_graph({("a", "b"): (100, 0)}, ["a", "b"], add_self_loops=False)
        assert g.edge_index.shape == (2, 1)
        assert g.edge_index[0, 0].item() == 0 and g.edge_index[1, 0].item() == 1

    def test_asymmetric_weights(self):
        # Same pair both ways with different traffic: weights must differ.
        g = build_call_graph(
            {("a", "b"): (100, 0), ("b", "a"): (100, 100), ("a", "c"): (1, 0)},
            ["a", "b", "c"], edge_alpha=0.5, add_self_loops=False,
        )
        w = {(g.services[i], g.services[j]): g.edge_weight[k].item()
             for k, (i, j) in enumerate(g.edge_index.t().tolist())}
        assert w[("a", "b")] != w[("b", "a")]

    def test_error_count_converted_to_rate(self):
        # 50 errors of 100 requests is rate 0.5, not raw count 50.
        g = build_call_graph(
            {("a", "b"): (100, 50), ("b", "c"): (100, 0)},
            ["a", "b", "c"], edge_alpha=0.0, add_self_loops=False,
        )
        # errors norm over rates [0.5, 0.0] -> [1, 0]
        assert np.allclose(sorted(g.edge_weight.tolist()),
                           sorted(sigmoid(np.array([1.0, 0.0])).tolist()))

    def test_zero_traffic_edge_dropped(self):
        g = build_call_graph({("a", "b"): (0, 0)}, ["a", "b"], add_self_loops=False)
        assert len(g) == 0

    def test_unknown_services_ignored(self):
        g = build_call_graph({("a", "zzz"): (10, 0)}, ["a", "b"], add_self_loops=False)
        assert len(g) == 0

    def test_self_loops_added_with_weight_one(self):
        g = build_call_graph({("a", "b"): (10, 0)}, ["a", "b", "c"], add_self_loops=True)
        loops = [(k, i) for k, (i, j) in enumerate(g.edge_index.t().tolist()) if i == j]
        assert len(loops) == 3
        assert all(g.edge_weight[k].item() == 1.0 for k, _ in loops)

    def test_node_ids_stable_across_windows(self):
        # Different traffic, same roster -> same node numbering. The model
        # depends on service index meaning the same thing in every window.
        services = ["a", "b", "c"]
        g1 = build_call_graph({("a", "b"): (10, 0)}, services)
        g2 = build_call_graph({("b", "c"): (10, 0)}, services)
        assert g1.services == g2.services == services

    def test_empty_window(self):
        g = build_call_graph({}, ["a", "b"], add_self_loops=False)
        assert len(g) == 0 and g.edge_index.shape == (2, 0)
