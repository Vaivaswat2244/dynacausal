#!/usr/bin/env python3
"""Phase 2 evaluation: run the trained Phase 1 model over live-cluster runs
WITHOUT the fault time (CLAUDE_CODE_TASK_phase2_eval.md, step 2).

For every prepared run, windows of the Phase 1 length slide over the whole
recording (stride = one 15 s bucket). Per window the model's score vector is
recorded; per-run results go to results/live/{service}_{fault}_{rep}.json.

Usage: python scripts/eval_live.py data/live/prepared [--config configs/d1.yaml]
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

import torch
import yaml

from src.data.dataset import build_sample
from src.data.graph import build_call_graph
from src.data.loaders import OB_SERVICES, Case
from src.data.preprocess import raw_case_features, scale_with_stats
from src.model.dynacausal import DynaCausal
from src.eval import evaluate

THRESHOLDS = [0.5, 0.7, 0.9]
N_METRIC_FEATS = 5  # metric families come first in FEATURE_NAMES


def window_at(sample, i0, T, edge_alpha, services=OB_SERVICES):
    """Window + graph at absolute bucket i0 (window_sample is inject-anchored)."""
    x = sample["feats"][:, i0:i0 + T, :]
    m = (sample["call_b"] >= i0) & (sample["call_b"] < i0 + T)
    stats = {}
    for s, d, n, e in zip(sample["call_src"][m].tolist(), sample["call_dst"][m].tolist(),
                          sample["call_n"][m].tolist(), sample["call_err"][m].tolist()):
        key = (services[s], services[d])
        pn, pe = stats.get(key, (0, 0))
        stats[key] = (pn + n, pe + e)
    g = build_call_graph(stats, list(services), edge_alpha=edge_alpha)
    return x, g.edge_index, g.edge_weight, stats


def eval_run(run_dir, model, T, edge_alpha, grid_s, stats=None):
    """Slide over one run; returns the per-window record list and timings.

    stats=None keeps the Phase 1 per-case min-max; otherwise (lo, hi) arrays
    fitted on D1 train replace it. Also returns the share of feature values
    that fall outside [0,1] after scaling and the load time in seconds.
    """
    t_load = time.perf_counter()
    case = Case.from_path(run_dir)
    if case.root_cause not in OB_SERVICES:
        # no-fault run ("none_none"): build_sample needs some valid label;
        # it is a placeholder and main() reports the label as None.
        case.root_cause = OB_SERVICES[0]
    sample = build_sample(case, grid_s)
    out_of_range = 0.0
    if stats is not None:
        _, raw = raw_case_features(case, grid_s)
        scaled = scale_with_stats(raw, *stats)
        out_of_range = float(((scaled < 0) | (scaled > 1)).mean())
        sample["feats"] = torch.tensor(scaled, dtype=torch.float32)
    t_load = time.perf_counter() - t_load
    meta = json.loads((run_dir / "run.json").read_text())
    n_win = sample["n_buckets"] - T + 1
    recs, lat = [], []
    for i0 in range(max(n_win, 1)):
        t0 = time.perf_counter()
        x, ei, ew, stats = window_at(sample, i0, T, edge_alpha)
        with torch.no_grad():
            scores = model(x.unsqueeze(0), ei, ew).squeeze(0)
        lat.append(time.perf_counter() - t0)

        # Baseline 1: metric-jump vs the previous window (first window: zeros).
        if i0 > 0:
            prev = sample["feats"][:, i0 - 1:i0 - 1 + T, :N_METRIC_FEATS]
            jump = (x[:, :, :N_METRIC_FEATS] - prev).abs().mean(dim=(1, 2))
        else:
            jump = torch.zeros(len(OB_SERVICES))
        # Baseline 2: errors on edges incident to the service.
        errc = torch.zeros(len(OB_SERVICES))
        for (a, b), (_, e) in stats.items():
            errc[OB_SERVICES.index(a)] += e
            errc[OB_SERVICES.index(b)] += e

        recs.append({
            "i0": i0,
            "start_unix": None,  # filled below from the grid origin
            "scores": [float(v) for v in scores],
            "jump": [float(v) for v in jump],
            "errc": [int(v) for v in errc],
        })
    # Bucket i starts at grid_origin + i*grid_s; reconstruct origin from inject.
    origin = meta["inject_time"] - sample["inject_idx"] * grid_s if meta["inject_time"] else meta["run_start"]
    for r in recs:
        r["start_unix"] = origin + r["i0"] * grid_s
    return sample, meta, recs, lat, t_load, out_of_range


def rank_of(scores, label):
    """Pessimistic rank (ties count against us), 1-based."""
    s = scores[label]
    return int(sum(1 for v in scores if v >= s))


def metrics_from_ranks(ranks):
    n = len(ranks)
    out = {f"AC@{k}": sum(r <= k for r in ranks) / n for k in (1, 3, 5)}
    out["MRR"] = sum(1 / r for r in ranks) / n
    out["n"] = n
    out["ranks"] = ranks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prepared_root")
    ap.add_argument("--config", default="configs/d1.yaml")
    ap.add_argument("--checkpoint", default="results/d1_full_model.pt")
    ap.add_argument("--norm", choices=["d1train", "percase"], default="d1train",
                    help="d1train: scale with results/d1_norm_stats.pt; "
                         "percase: Phase 1 per-recording min-max")
    ap.add_argument("--stats", default="results/d1_norm_stats.pt")
    ap.add_argument("--out", default="results/live")
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

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in Path(a.prepared_root).glob("*_*/*/run.json"))
    per_run, all_lat, loads, oors = [], [], [], []
    for rd in runs:
        svc_fault = rd.parent.name
        sample, meta, recs, lat, t_load, oor = eval_run(rd, model, T, alpha, grid_s, stats)
        all_lat += lat; loads.append(t_load); oors.append(oor)
        rec = {"run": f"{svc_fault}/{rd.name}", "meta": meta,
               "label": sample["label"] if meta["inject_time"] else None,
               "inject_idx": sample["inject_idx"], "n_buckets": sample["n_buckets"],
               "grid_s": grid_s, "T": T, "norm": a.norm,
               "share_feature_values_outside_0_1": oor, "windows": recs}
        per_run.append(rec)
        svc, fault = svc_fault.rsplit("_", 1)
        (out_dir / f"{svc}_{fault}_{rd.name}.json").write_text(json.dumps(rec))
        tops = [max(range(len(w["scores"])), key=lambda i: w["scores"][i]) for w in recs]
        print(f"{rec['run']}: {len(recs)} windows, inject_idx {rec['inject_idx']}, "
              f"label {rec['label']}, top-1 per window {tops}")

    fault_runs = [r for r in per_run if r["meta"]["inject_time"]]
    none_runs = [r for r in per_run if not r["meta"]["inject_time"]]

    def window_near_inject(r):
        i = min(r["inject_idx"], len(r["windows"]) - 1)
        return r["windows"][i]

    report = {}
    # 1. accuracy at fault time; model and baselines on the same window
    for key, vec in [("model", "scores"), ("metric_jump", "jump"), ("error_count", "errc")]:
        ranks = [rank_of(window_near_inject(r)[vec], r["label"]) for r in fault_runs]
        report.setdefault("at_fault_time", {})[key] = metrics_from_ranks(ranks)
    # 2. model's own choice of when: window with the highest top score
    for key, vec in [("model", "scores"), ("metric_jump", "jump"), ("error_count", "errc")]:
        ranks = []
        for r in fault_runs:
            w = max(r["windows"], key=lambda w: max(w[vec]))
            ranks.append(rank_of(w[vec], r["label"]))
        report.setdefault("no_fault_time", {})[key] = metrics_from_ranks(ranks)
    # 3. time to diagnosis (model only)
    ttds, never = [], 0
    for r in fault_runs:
        inj, dur = r["meta"]["inject_time"], r["meta"]["fault_end"] - r["meta"]["inject_time"]
        hit = next((w["start_unix"] - inj for w in r["windows"]
                    if w["start_unix"] >= inj and max(range(len(w["scores"])),
                       key=lambda i: w["scores"][i]) == r["label"]), None)
        if hit is None or hit > dur:
            never += 1
        else:
            ttds.append(hit)
    ttds.sort()
    report["time_to_diagnosis"] = {
        "median_s": statistics.median(ttds) if ttds else None, "all_s": ttds,
        "diagnosed": len(ttds), "never_within_fault": never, "n": len(fault_runs)}
    # 4. false alarms vs detection, by top-score threshold
    def share_over(wins, thr):
        return sum(max(w["scores"]) > thr for w in wins) / len(wins) if wins else None
    none_w = [w for r in none_runs for w in r["windows"]]
    pre_w = [w for r in fault_runs for w in r["windows"] if w["start_unix"] + T * grid_s <= r["meta"]["inject_time"]]
    post_w = [w for r in fault_runs for w in r["windows"]
              if r["meta"]["inject_time"] <= w["start_unix"] < r["meta"]["fault_end"]]
    report["false_alarms"] = {str(t): {"none_runs": share_over(none_w, t),
                                       "pre_fault": share_over(pre_w, t),
                                       "post_fault": share_over(post_w, t)} for t in THRESHOLDS}
    report["n_windows"] = {"none": len(none_w), "pre": len(pre_w), "post": len(post_w)}
    def top_stats(wins):
        t = sorted(max(w["scores"]) for w in wins)
        return {"min": t[0], "median": statistics.median(t), "max": t[-1]} if t else None
    report["top_score"] = {"none_runs": top_stats(none_w), "pre_fault": top_stats(pre_w),
                           "post_fault": top_stats(post_w)}
    # 5. per fault type
    for r in fault_runs:
        ft = r["run"].split("/")[0].rsplit("_", 1)[1]
        report.setdefault("per_fault", {}).setdefault(ft, {"at": [], "no": []})
        report["per_fault"][ft]["at"].append(rank_of(window_near_inject(r)["scores"], r["label"]))
        w = max(r["windows"], key=lambda w: max(w["scores"]))
        report["per_fault"][ft]["no"].append(rank_of(w["scores"], r["label"]))
    for ft, d in report.get("per_fault", {}).items():
        report["per_fault"][ft] = {"at_fault_time": metrics_from_ranks(d["at"]),
                                   "no_fault_time": metrics_from_ranks(d["no"])}
    # 6. latency and memory
    all_lat.sort()
    report["pipeline"] = {
        "per_window_ms_median": statistics.median(all_lat) * 1000,
        "per_window_ms_p95": all_lat[int(len(all_lat) * 0.95)] * 1000,
        "load_features_per_run_s_median": statistics.median(loads),
        "load_features_per_run_s_max": max(loads),
        "n_windows_total": len(all_lat),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    report["norm"] = a.norm
    report["share_feature_values_outside_0_1"] = {"mean": sum(oors) / len(oors), "max": max(oors)}
    report["n_runs"] = {"fault": len(fault_runs), "none": len(none_runs)}

    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
