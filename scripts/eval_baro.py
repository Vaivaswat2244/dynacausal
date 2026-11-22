#!/usr/bin/env python3
"""BARO baseline (Pham et al., FSE 2024; pip package fse-baro) on live runs.

Two modes per run, on the run's raw metrics.csv:
  inject  BARO's RobustScorer told the fault time (run.json inject_time).
  cpd     BARO's own multivariate BOCPD picks the change point; RobustScorer
          then splits the data there. No fault time is given.

The package's code is used as is. It returns a ranked list of METRIC columns
without scores, so two thin steps are added here: a column is mapped to the
service whose name it starts with (first appearance in the ranking decides
the service's position), and a service's score is written as
1 / position so the shared metric code can rank it. Services BARO did not
rank at all get score 0, and ties count against the method.

Output: <out>/baro_inject/ and <out>/baro_cpd/, one JSON per run in the same
format as eval_live.py.

BOCPD over all metric columns takes minutes per run, so runs are spread over
--jobs processes and a run whose output file already exists is not redone.

Usage: python scripts/eval_baro.py data/live/<batch> [--out results/live_ext/<batch>] [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one core per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer

from src.data.live import LIVE_RENAME
from src.data.loaders import OB_SERVICES
from src.live_report import full_report

GRID_S, T = 15, 20   # only used so the shared report code can read the record


def service_scores(ranked_cols) -> list[float]:
    """Ranked metric columns -> one score per service (1 / first position)."""
    order = []
    for col in ranked_cols:
        svc = col.split("_", 1)[0]
        svc = LIVE_RENAME.get(svc, svc)
        if svc in OB_SERVICES and svc not in order:
            order.append(svc)
    return [1.0 / (order.index(s) + 1) if s in order else 0.0 for s in OB_SERVICES]


def run_baro(run_dir: Path, mode: str) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    data = pd.read_csv(run_dir / "metrics.csv")
    label = OB_SERVICES.index(meta["service"]) if meta.get("inject_time") else None
    rec = {"run": f"{run_dir.parent.name}/{run_dir.name}", "meta": meta, "label": label,
           "inject_idx": None, "grid_s": GRID_S, "T": T, "method": f"baro_{mode}",
           "single": "at" if mode == "inject" else "free",
           "services": list(OB_SERVICES), "windows": [], "note": None}
    t0 = time.perf_counter()
    try:
        if mode == "inject":
            if not meta.get("inject_time"):
                rec["note"] = "no-fault run: nothing to rank in inject mode"
                return rec
            out = robust_scorer(data, inject_time=meta["inject_time"])
            when = meta["inject_time"]
        else:
            cps = bocpd(data)
            rec["change_points"] = [int(c) for c in cps]
            if not cps:
                rec["note"] = "BOCPD found no change point"
                return rec
            out = robust_scorer(data, anomalies=cps)
            when = int(data["time"].iloc[cps[0]])
        rec["windows"] = [{"i0": 0, "start_unix": when, "scores": service_scores(out["ranks"]),
                           "top_metrics": out["ranks"][:5]}]
    except Exception as e:  # report, do not hide
        rec["note"] = f"BARO raised {type(e).__name__}: {e}"
    finally:
        rec["seconds"] = time.perf_counter() - t0
    return rec


def _job(args):
    """One (run, mode) in a worker; reuses the output file when it exists."""
    rd, mode, out_file = args
    if out_file.exists():
        return json.loads(out_file.read_text())
    rec = run_baro(rd, mode)
    out_file.write_text(json.dumps(rec))
    print(f"{mode} {rec['run']}: {rec['seconds']:.1f}s "
          f"{rec.get('change_points', '')} {rec['note'] or ''}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    root = Path(a.runs_root)
    out_root = Path(a.out or f"results/live_ext/{root.name}")
    jobs = []
    for mode in ("inject", "cpd"):
        (out_root / f"baro_{mode}").mkdir(parents=True, exist_ok=True)
        for rd in sorted(p.parent for p in root.glob("*_*/*/run.json")):
            svc, fault = rd.parent.name.rsplit("_", 1)
            jobs.append((rd, mode, out_root / f"baro_{mode}" / f"{svc}_{fault}_{rd.name}.json"))
    with ProcessPoolExecutor(a.jobs) as ex:
        recs = list(ex.map(_job, jobs))
    for mode in ("inject", "cpd"):
        mine = [r for r in recs if r["method"] == f"baro_{mode}"]
        rep = full_report(mine, "scores", windowed=False)
        rep["seconds_per_run_median"] = sorted(r["seconds"] for r in mine)[len(mine) // 2]
        rep["runs_without_ranking"] = [r["run"] for r in mine if not r["windows"]]
        (out_root / f"baro_{mode}" / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
