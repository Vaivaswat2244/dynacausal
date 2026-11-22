#!/usr/bin/env python3
"""Build the markdown tables for one batch from the per-run ranking files.

Reads results/live_ext/<batch>/<method>/*.json (written by eval_live.py and
eval_baro.py) and prints every table twice: all fault runs, and impactful
runs only. Numbers are shown to 4 decimals; the unrounded values are in each
method's report.json.

Usage: python scripts/report_live.py results/live_ext/<batch> [> tables.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.live_report import full_report, is_fault, is_impactful, load_method

# display name -> (folder, vector key, windowed, scores are probabilities)
def methods_for(root: Path, main="dynacausal_causal"):
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name.startswith("dynacausal"):
            out[d.name] = (d, "scores", True, True)
        elif d.name.startswith("baro"):
            out[d.name] = (d, "scores", False, False)
    if (root / main).is_dir():
        out["metric_jump"] = (root / main, "jump", True, False)
        out["error_count"] = (root / main, "errc", True, False)
    return out


def f(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


ACC = ["AC@1", "AC@3", "AC@5", "Avg@5", "MRR", "n"]


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def render(root: Path) -> str:
    meths = methods_for(root)
    reps = {name: full_report(load_method(d), vec, windowed=w, scores_are_probabilities=p)
            for name, (d, vec, w, p) in meths.items()}
    any_recs = load_method(next(iter(meths.values()))[0])
    fault = [r for r in any_recs if is_fault(r)]
    md = [f"Runs: {len(fault)} fault runs, of which {sum(is_impactful(r) for r in fault)} impactful; "
          f"{len(any_recs) - len(fault)} no-fault runs.\n"]
    for sub in ("all", "impactful"):
        n = next(iter(reps.values()))[sub]["n_fault_runs"]
        md.append(f"#### Subset: {sub} fault runs (n={n})\n")
        for key, title in (("at_fault_time", "1. Accuracy at fault time (window starting at inject)"),
                           ("no_fault_time", "2. Accuracy without fault time (method's own choice of when)")):
            rows = [[m] + [r[sub]["accuracy"][key][c] for c in ACC] for m, r in reps.items()
                    if not (key == "at_fault_time" and m == "baro_cpd")
                    and not (key == "no_fault_time" and m == "baro_inject")]
            md.append(f"**{title}**\n\n" + table(["method"] + ACC, rows))
        rows = [[m, t["median_s"], t["diagnosed"], t["never_within_fault"], t["share_never"], t["n"]]
                for m, r in reps.items() if (t := r[sub].get("time_to_diagnosis"))]
        md.append("**3. Time to diagnosis**\n\n" + table(
            ["method", "median s (diagnosed runs)", "diagnosed", "never within fault", "share never", "n"], rows))
        rows = []
        for m, r in reps.items():
            fa = r[sub].get("false_alarms")
            if fa:
                nw = fa["n_windows"]
                for t in ("0.5", "0.7", "0.9"):
                    rows.append([m, t, fa[t]["no_fault_runs"], nw["no_fault_runs"], fa[t]["pre_fault"],
                                 nw["pre_fault"], fa[t]["during_fault"], nw["during_fault"]])
        md.append("**4. Share of windows with top score above threshold**\n\n" + table(
            ["method", "threshold", "no-fault runs", "n win", "pre-fault", "n win", "during fault", "n win"], rows))
        for grp, title in (("per_fault", "5a. Per fault type"), ("per_service", "5b. Per service")):
            rows = []
            for m, r in reps.items():
                for g, acc in r[sub][grp].items():
                    a, b = acc["at_fault_time"], acc["no_fault_time"]
                    rows.append([m, g, a["n"], a["AC@1"], a["AC@3"], a["AC@5"], a["MRR"],
                                 b["AC@1"], b["AC@3"], b["AC@5"], b["MRR"]])
            md.append(f"**{title}** (at = at fault time, free = without fault time; baro_inject has "
                      "only at, baro_cpd only free, the other side counts every run as last)\n\n" + table(
                ["method", "group", "n", "at AC@1", "at AC@3", "at AC@5", "at MRR",
                 "free AC@1", "free AC@3", "free AC@5", "free MRR"], rows))
    rows = []
    for m, (d, vec, w, p) in meths.items():
        rp = d / "report.json"
        if vec == "scores" and rp.exists():
            j = json.loads(rp.read_text())
            if "pipeline" in j:
                q = j["pipeline"]
                rows.append([m, q["per_window_ms_median"], q["per_window_ms_p95"],
                             q["load_features_per_run_s_median"], q["n_windows_scored"],
                             q["n_windows_skipped"], q["peak_rss_mb"],
                             j["share_feature_values_outside_0_1"]["mean"]])
            elif "seconds_per_run_median" in j:
                rows.append([m, "n/a", "n/a", j["seconds_per_run_median"], "n/a", "n/a", "n/a", "n/a"])
    md.append("**9. Pipeline** (per window = scaling + graph + forward pass; load = reading and "
              "bucketing one run's files; for BARO the load column is its whole run time in s)\n\n" + table(
        ["method", "ms/window median", "ms/window p95", "load s/run median", "windows scored",
         "windows skipped", "peak RSS MB", "share of inputs outside 0..1"], rows))
    return "\n".join(md)


if __name__ == "__main__":
    print(render(Path(sys.argv[1])))
