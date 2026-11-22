#!/usr/bin/env python3
"""Phase 2 evaluation: run the trained Phase 1 model over live-cluster runs
WITHOUT the fault time.

For every prepared run, windows of the Phase 1 length slide over the whole
recording (stride = one 15 s bucket). Per window the model's score vector
and the two simple baselines are recorded; per-run results go to
<out>/dynacausal_<norm>/{service}_{fault}_{rep}.json and a summary to
report.json in the same folder.

Scaling (--norm):
  causal      default. Each window is scaled with the min/max of the run up
              to the window's first bucket only, clipped to [0,1]. Windows
              with less than 120 s of history are skipped and counted.
  causal_raw  same history rule, no clipping (shown once for comparison).
  d1train     fixed min/max fitted on the D1 train cases.
  percase     Phase 1 rule: min/max over the whole recording (uses the future).

Usage: python scripts/eval_live.py data/live/prepared/<batch> [--norm causal]
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.data.dataset import build_sample
from src.data.graph import build_call_graph
from src.data.loaders import OB_SERVICES, Case
from src.data.preprocess import (causal_scale, min_max_per_series,
                                 raw_case_features, scale_with_stats)
from src.live_report import full_report
from src.model.dynacausal import DynaCausal

N_METRIC_FEATS = 5       # metric families come first in FEATURE_NAMES
MIN_HISTORY_S = 120      # causal scaling needs at least this much past data


def window_graph(sample, i0, T, edge_alpha, services=OB_SERVICES):
    """Call graph and per-edge (requests, errors) for buckets [i0, i0+T)."""
    m = (sample["call_b"] >= i0) & (sample["call_b"] < i0 + T)
    stats = {}
    for s, d, n, e in zip(sample["call_src"][m].tolist(), sample["call_dst"][m].tolist(),
                          sample["call_n"][m].tolist(), sample["call_err"][m].tolist()):
        key = (services[s], services[d])
        pn, pe = stats.get(key, (0, 0))
        stats[key] = (pn + n, pe + e)
    g = build_call_graph(stats, list(services), edge_alpha=edge_alpha)
    return g.edge_index, g.edge_weight, stats


def make_scaler(norm, raw, stats):
    """Returns f(i_start, i_end, i0) -> scaled (N, i_end-i_start, D) array.

    i0 is the first bucket of the window being scored; for the causal modes
    the scaling statistics come from buckets [0, i0) only.
    """
    if norm == "percase":
        full = min_max_per_series(raw)
        return lambda a, b, i0: full[:, a:b]
    if norm == "d1train":
        full = scale_with_stats(raw, *stats)
        return lambda a, b, i0: full[:, a:b]
    clip = norm == "causal"
    return lambda a, b, i0: causal_scale(raw[:, a:b], raw[:, :i0], clip=clip)


def eval_run(run_dir, model, T, edge_alpha, grid_s, norm, stats=None):
    """Slide over one run. Returns the per-run record and timings."""
    t_load = time.perf_counter()
    case = Case.from_path(run_dir)
    if case.root_cause not in OB_SERVICES:
        # no-fault run ("none_none"): build_sample needs some valid label;
        # it is a placeholder and the record's label is None.
        case.root_cause = OB_SERVICES[0]
    sample = build_sample(case, grid_s)          # call stats, inject index
    _, raw = raw_case_features(case, grid_s)     # un-normalised (N, G, D)
    scale = make_scaler(norm, raw, stats)
    t_load = time.perf_counter() - t_load
    meta = json.loads((run_dir / "run.json").read_text())
    origin = meta["inject_time"] - sample["inject_idx"] * grid_s if meta["inject_time"] else meta["run_start"]

    n_win = max(sample["n_buckets"] - T + 1, 1)
    min_i0 = -(-MIN_HISTORY_S // grid_s) if norm.startswith("causal") else 0
    recs, lat, skipped, outside = [], [], 0, []
    for i0 in range(n_win):
        if i0 < min_i0:
            skipped += 1
            continue
        t0 = time.perf_counter()
        x_np = scale(i0, i0 + T, i0)
        x = torch.tensor(x_np, dtype=torch.float32)
        ei, ew, edge_stats = window_graph(sample, i0, T, edge_alpha)
        with torch.no_grad():
            scores = model(x.unsqueeze(0), ei, ew).squeeze(0)
        lat.append(time.perf_counter() - t0)
        outside.append(float(((x_np < 0) | (x_np > 1)).mean()))

        # Baseline 1: metric-jump. Mean absolute change of the service's own
        # metric features between this window and the one a bucket earlier,
        # both scaled the same way as this window.
        if i0 > 0:
            prev = scale(i0 - 1, i0 - 1 + T, i0)[:, :, :N_METRIC_FEATS]
            jump = np.abs(x_np[:, :, :N_METRIC_FEATS] - prev).mean(axis=(1, 2))
        else:
            jump = np.zeros(len(OB_SERVICES))
        # Baseline 2: failed calls on edges touching the service.
        errc = [0] * len(OB_SERVICES)
        for (a, b), (_, e) in edge_stats.items():
            errc[OB_SERVICES.index(a)] += e
            errc[OB_SERVICES.index(b)] += e

        recs.append({"i0": i0, "start_unix": origin + i0 * grid_s,
                     "scores": [float(v) for v in scores],
                     "jump": [float(v) for v in jump], "errc": errc})

    rec = {"run": f"{run_dir.parent.name}/{run_dir.name}", "meta": meta,
           "label": sample["label"] if meta["inject_time"] else None,
           "inject_idx": sample["inject_idx"], "n_buckets": sample["n_buckets"],
           "grid_s": grid_s, "T": T, "method": f"dynacausal_{norm}", "single": None,
           "services": list(OB_SERVICES), "skipped_windows": skipped,
           "share_feature_values_outside_0_1": sum(outside) / len(outside) if outside else None,
           "windows": recs}
    return rec, lat, t_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_root")
    ap.add_argument("--config", default="configs/d1.yaml")
    ap.add_argument("--checkpoint", default="results/d1_full_model.pt")
    ap.add_argument("--norm", choices=["causal", "causal_raw", "d1train", "percase"], default="causal")
    ap.add_argument("--stats", default="results/d1_norm_stats.pt")
    ap.add_argument("--out", default=None, help="default results/live_ext/<batch>")
    ap.add_argument("--tag", default=None, help="method folder name, default dynacausal_<norm>")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    T, alpha, grid_s = cfg["data"]["window_T"], cfg["data"]["edge_alpha"], cfg["data"]["grid_s"]
    model = DynaCausal(**cfg["model"])
    model.load_state_dict(torch.load(a.checkpoint, weights_only=True))
    model.eval()

    stats = None
    if a.norm == "d1train":
        st = torch.load(a.stats, weights_only=True)
        assert st["services"] == list(OB_SERVICES)
        stats = (st["lo"].numpy().astype("float64"), st["hi"].numpy().astype("float64"))

    root = Path(a.prepared_root)
    out_dir = Path(a.out or f"results/live_ext/{root.name}") / (a.tag or f"dynacausal_{a.norm}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run, all_lat, loads = [], [], []
    for rd in sorted(p.parent for p in root.glob("*_*/*/run.json")):
        rec, lat, t_load = eval_run(rd, model, T, alpha, grid_s, a.norm, stats)
        all_lat += lat; loads.append(t_load)
        per_run.append(rec)
        svc, fault = rd.parent.name.rsplit("_", 1)
        (out_dir / f"{svc}_{fault}_{rd.name}.json").write_text(json.dumps(rec))
        print(f"{rec['run']}: {len(rec['windows'])} windows, {rec['skipped_windows']} skipped, "
              f"inject_idx {rec['inject_idx']}, label {rec['label']}")

    report = {"norm": a.norm, "checkpoint": a.checkpoint,
              "model": full_report(per_run, "scores", scores_are_probabilities=True),
              "metric_jump": full_report(per_run, "jump"),
              "error_count": full_report(per_run, "errc")}
    all_lat.sort()
    oor = [r["share_feature_values_outside_0_1"] for r in per_run if r["windows"]]
    report["pipeline"] = {
        "per_window_ms_median": statistics.median(all_lat) * 1000,
        "per_window_ms_p95": all_lat[int(len(all_lat) * 0.95)] * 1000,
        "load_features_per_run_s_median": statistics.median(loads),
        "load_features_per_run_s_max": max(loads),
        "n_windows_scored": len(all_lat),
        "n_windows_skipped": sum(r["skipped_windows"] for r in per_run),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    report["share_feature_values_outside_0_1"] = {"mean": sum(oor) / len(oor), "max": max(oor)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    acc = report["model"]
    print(json.dumps({k: {"n": acc[k]["n_fault_runs"], **acc[k]["accuracy"]} for k in ("all", "impactful")}, indent=1))


if __name__ == "__main__":
    main()
