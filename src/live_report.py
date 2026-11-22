"""Shared metric code for the live-cluster evaluation.

Every method (DynaCausal, BARO, the simple baselines) writes one JSON per
run with the same shape:

    {"run": "svc_fault/rep", "meta": <run.json>, "label": int or None,
     "inject_idx": int, "grid_s": int, "T": int,
     "single": None | "at" | "free",
     "windows": [{"i0": int, "start_unix": int, "<vec>": [12 numbers]}, ...]}

"single" marks methods that give one ranking per run instead of one per
window: "at" = made knowing the inject time, "free" = made at the method's
own change point. Everything here works from those files, so the numbers
for all methods come from the same code.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

THRESHOLDS = [0.5, 0.7, 0.9]


def rank_of(scores, label) -> int:
    """Pessimistic 1-based rank: ties count against the method."""
    s = scores[label]
    return int(sum(1 for v in scores if v >= s))


def metrics_from_ranks(ranks) -> dict:
    """AC@1/3/5, Avg@5 (mean of AC@1..5, the paper's definition), MRR, n."""
    n = len(ranks)
    if n == 0:
        return {"AC@1": None, "AC@3": None, "AC@5": None, "Avg@5": None, "MRR": None, "n": 0}
    ac = {k: sum(r <= k for r in ranks) / n for k in range(1, 6)}
    return {"AC@1": ac[1], "AC@3": ac[3], "AC@5": ac[5],
            "Avg@5": sum(ac.values()) / 5, "MRR": sum(1 / r for r in ranks) / n, "n": n}


def load_method(method_dir) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(method_dir).glob("*.json"))
            if p.name != "report.json"]


def is_fault(r) -> bool:
    return bool(r["meta"].get("inject_time"))


def is_impactful(r) -> bool:
    imp = r["meta"].get("impact")
    return bool(imp and imp.get("impactful"))


def window_at_inject(r):
    """The window starting at the inject bucket (clamped to the last window)."""
    if r.get("single") == "at":
        return r["windows"][0] if r["windows"] else None
    if r.get("single") == "free" or not r["windows"]:
        return None
    target = min(r["inject_idx"], r["windows"][-1]["i0"])
    return next((w for w in r["windows"] if w["i0"] == target), None)


def window_free(r, vec):
    """The method's own choice of when: the window with the highest top value."""
    if r.get("single") == "free":
        return r["windows"][0] if r["windows"] else None
    if r.get("single") == "at" or not r["windows"]:
        return None
    return max(r["windows"], key=lambda w: max(w[vec]))


def ranks(recs, vec, pick):
    """Ranks of the true service; a run with no usable window counts as last."""
    out = []
    for r in recs:
        w = pick(r)
        out.append(rank_of(w[vec], r["label"]) if w is not None else len(r["services"]))
    return out


def accuracy(recs, vec) -> dict:
    return {"at_fault_time": metrics_from_ranks(ranks(recs, vec, window_at_inject)),
            "no_fault_time": metrics_from_ranks(ranks(recs, vec, lambda r: window_free(r, vec)))}


def time_to_diagnosis(recs, vec) -> dict:
    """First window start at or after inject whose top-1 is the true service."""
    ttds, never = [], 0
    for r in recs:
        inj, dur = r["meta"]["inject_time"], r["meta"]["fault_end"] - r["meta"]["inject_time"]
        hit = next((w["start_unix"] - inj for w in r["windows"]
                    if w["start_unix"] >= inj and rank_of(w[vec], r["label"]) == 1), None)
        if hit is None or hit > dur:
            never += 1
        else:
            ttds.append(hit)
    n = len(recs)
    return {"median_s": statistics.median(ttds) if ttds else None, "diagnosed": len(ttds),
            "never_within_fault": never, "share_never": never / n if n else None, "n": n}


def false_alarms(fault_recs, none_recs, vec) -> dict:
    """Share of windows whose top score exceeds each threshold, by period."""
    def share(wins, thr):
        return sum(max(w[vec]) > thr for w in wins) / len(wins) if wins else None
    span = lambda r: r["T"] * r["grid_s"]
    none_w = [w for r in none_recs for w in r["windows"]]
    pre_w = [w for r in fault_recs for w in r["windows"]
             if w["start_unix"] + span(r) <= r["meta"]["inject_time"]]
    dur_w = [w for r in fault_recs for w in r["windows"]
             if r["meta"]["inject_time"] <= w["start_unix"] < r["meta"]["fault_end"]]
    out = {str(t): {"no_fault_runs": share(none_w, t), "pre_fault": share(pre_w, t),
                    "during_fault": share(dur_w, t)} for t in THRESHOLDS}
    out["n_windows"] = {"no_fault_runs": len(none_w), "pre_fault": len(pre_w), "during_fault": len(dur_w)}
    return out


def breakdown(recs, vec, key) -> dict:
    """accuracy() per group; key(r) gives the group name."""
    groups = {}
    for r in recs:
        groups.setdefault(key(r), []).append(r)
    return {g: accuracy(rs, vec) for g, rs in sorted(groups.items())}


def fault_of(r) -> str:
    return r["run"].split("/")[0].rsplit("_", 1)[1]


def service_of(r) -> str:
    return r["run"].split("/")[0].rsplit("_", 1)[0]


def full_report(recs, vec, windowed=True, scores_are_probabilities=False) -> dict:
    """Everything for one method, twice: all fault runs and impactful only."""
    none_recs = [r for r in recs if not is_fault(r)]
    out = {}
    for name, sel in (("all", lambda r: True), ("impactful", is_impactful)):
        fr = [r for r in recs if is_fault(r) and sel(r)]
        d = {"n_fault_runs": len(fr), "accuracy": accuracy(fr, vec),
             "per_fault": breakdown(fr, vec, fault_of),
             "per_service": breakdown(fr, vec, service_of)}
        if windowed:
            d["time_to_diagnosis"] = time_to_diagnosis(fr, vec)
        if scores_are_probabilities:
            d["false_alarms"] = false_alarms(fr, none_recs, vec)
        out[name] = d
    out["n_no_fault_runs"] = len(none_recs)
    return out
