"""Build the trainable D1 dataset.

Stores, per case, the FULL feature series and per-bucket call statistics,
not a pre-cut window. Window length, window position and edge_alpha are then
applied at load time (see window_sample), so tuning any of them never touches
the raw CSVs again.

Split (from the brief): stratified by (service, fault) with no case on both
sides. Repetitions 1,2 -> train, 3 -> test: exactly 60/30, deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import build_call_graph
from .loaders import OB_SERVICES, Case, discover_cases, load_client_calls
from .preprocess import case_features


def build_sample(case: Case, grid_s: int, services=OB_SERVICES) -> dict:
    """One case -> full series + bucketed call stats + label."""
    grid, feats = case_features(case, grid_s=grid_s, services=services)

    traces = pd.read_csv(case.path / "traces.csv",
                         usecols=["serviceName", "operationName", "startTime", "statusCode"])
    calls = load_client_calls(traces)
    sidx = {s: i for i, s in enumerate(services)}
    buckets = ((calls["sec"] - grid[0]) // grid_s).astype(int)
    agg = (pd.DataFrame({
        "b": buckets,
        "src": calls["caller"].map(sidx),
        "dst": calls["callee"].map(sidx),
        "err": calls["error"].astype(int),
    }).dropna().astype(int).groupby(["b", "src", "dst"])["err"].agg(["size", "sum"]))

    return {
        "feats": torch.tensor(feats, dtype=torch.float32),      # (N, G, D)
        "inject_idx": int((grid >= case.inject_time).argmax()),
        "n_buckets": len(grid),
        # parallel arrays: calls on edge (src->dst) during bucket b
        "call_b": torch.tensor(agg.index.get_level_values("b").to_numpy()),
        "call_src": torch.tensor(agg.index.get_level_values("src").to_numpy()),
        "call_dst": torch.tensor(agg.index.get_level_values("dst").to_numpy()),
        "call_n": torch.tensor(agg["size"].to_numpy()),
        "call_err": torch.tensor(agg["sum"].to_numpy()),
        "label": services.index(case.root_cause),
        "case": f"{case.root_cause}_{case.fault}/{case.rep}",
        "split": "train" if case.rep in (1, 2) else "test",
    }


def window_sample(sample: dict, window_T: int, edge_alpha: float,
                  services=OB_SERVICES, offset: int = 0):
    """Cut the anomaly window, the pre-fault normal window, and the window's graph.

    Returns (x, x_norm, edge_index, edge_weight).
    - anomaly window: T buckets from injection (+offset), clamped to the series.
    - normal window:  the T buckets immediately before injection. Every case
      records 12 fault-free minutes first, so this always exists for T <= 48.
    """
    i0 = min(sample["inject_idx"] + offset, sample["n_buckets"] - window_T)
    x = sample["feats"][:, i0:i0 + window_T, :]
    n0 = max(sample["inject_idx"] - window_T, 0)
    x_norm = sample["feats"][:, n0:n0 + window_T, :]

    m = (sample["call_b"] >= i0) & (sample["call_b"] < i0 + window_T)
    stats = {}
    for s, d, n, e in zip(sample["call_src"][m].tolist(), sample["call_dst"][m].tolist(),
                          sample["call_n"][m].tolist(), sample["call_err"][m].tolist()):
        key = (services[s], services[d])
        pn, pe = stats.get(key, (0, 0))
        stats[key] = (pn + n, pe + e)
    g = build_call_graph(stats, list(services), edge_alpha=edge_alpha)
    return x, x_norm, g.edge_index, g.edge_weight


def build_dataset(raw_root: str, out_path: str, grid_s: int = 15) -> dict:
    cases = [c for c in discover_cases(raw_root) if c.root_cause in OB_SERVICES]
    samples = []
    for i, case in enumerate(cases):
        samples.append(build_sample(case, grid_s))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(cases)}", flush=True)
    data = {"samples": samples, "meta": {"grid_s": grid_s, "services": list(OB_SERVICES)}}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    n_train = sum(s["split"] == "train" for s in samples)
    print(f"saved {len(samples)} samples ({n_train} train / {len(samples) - n_train} test) -> {out_path}")
    return data
