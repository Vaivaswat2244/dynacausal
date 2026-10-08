"""Build the trainable D1 dataset: one (features, graph, label) sample per case.

Window choice (open decision 4-adjacent; paper silent): the T buckets starting
at the injection time. That is the window where the fault is active from the
start, the natural analogue of "the incident window" an operator would examine.

Split (from the brief): stratified by (service, fault) with no case in both
sides. Each of the 30 groups has repetitions 1,2,3 -> reps 1 and 2 train,
rep 3 test. Exactly 60/30, deterministic, perfectly stratified.
"""
from __future__ import annotations

from pathlib import Path

import torch

from .graph import build_call_graph
from .loaders import OB_SERVICES, Case, discover_cases, load_client_calls, window_call_stats
from .preprocess import case_features

import pandas as pd


def build_sample(case: Case, grid_s: int, window_T: int, edge_alpha: float,
                 services=OB_SERVICES) -> dict:
    grid, feats = case_features(case, grid_s=grid_s, services=services)

    # Window = T buckets from the injection bucket (clamped to the recording).
    start_idx = int((grid >= case.inject_time).argmax())
    start_idx = min(start_idx, len(grid) - window_T)
    sel = slice(start_idx, start_idx + window_T)
    x = torch.tensor(feats[:, sel, :], dtype=torch.float32)   # (N, T, D)

    t0, t1 = int(grid[sel.start]), int(grid[sel.stop - 1]) + grid_s
    traces = pd.read_csv(case.path / "traces.csv",
                         usecols=["serviceName", "operationName", "startTime", "statusCode"])
    calls = load_client_calls(traces)
    g = build_call_graph(window_call_stats(calls, t0, t1 - t0), list(services),
                         edge_alpha=edge_alpha)

    return {
        "x": x,
        "edge_index": g.edge_index,
        "edge_weight": g.edge_weight,
        "label": services.index(case.root_cause),
        "case": f"{case.root_cause}_{case.fault}/{case.rep}",
        "split": "train" if case.rep in (1, 2) else "test",
    }


def build_dataset(raw_root: str, out_path: str, grid_s: int = 15,
                  window_T: int = 20, edge_alpha: float = 0.5) -> dict:
    cases = discover_cases(raw_root)
    # The archive contains stray entries beyond the 90 real cases; keep only
    # directories whose name parses to a known Online Boutique service.
    kept = [c for c in cases if c.root_cause in OB_SERVICES]
    if len(kept) != len(cases):
        skipped = [str(c.path) for c in cases if c not in kept]
        print(f"skipping {len(skipped)} non-case path(s): {skipped}")
    cases = kept
    samples = []
    for i, case in enumerate(cases):
        samples.append(build_sample(case, grid_s, window_T, edge_alpha))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(cases)} cases")
    meta = {"grid_s": grid_s, "window_T": window_T, "edge_alpha": edge_alpha,
            "services": list(OB_SERVICES)}
    data = {"samples": samples, "meta": meta}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    n_train = sum(s["split"] == "train" for s in samples)
    print(f"saved {len(samples)} samples ({n_train} train / {len(samples) - n_train} test) -> {out_path}")
    return data
