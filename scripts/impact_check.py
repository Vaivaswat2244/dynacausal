#!/usr/bin/env python3
"""Did the fault reach the user? Frontend latency and errors, before vs during.

For a run, take the frontend's own HTTP entry spans (serviceName frontend,
operation without a "/", e.g. GET / POST) from traces.csv and compare
[inject-150, inject) with [inject, fault_end):
  - p95 duration (microseconds, as stored)
  - share of those spans with a non-zero status code
The run is "impactful" if p95 rose by at least 50% or the error share rose
by at least 0.05 absolute. If the frontend produced no entry spans at all
during the fault (but did before), the run is also flagged impactful, with
reason "no_frontend_spans_during_fault", since no ratio can be computed.

Writes the result into run.json under "impact" (null for no-fault runs).
Needs only pandas, so it also runs on the recording instance.

Usage: impact_check.py RUN_DIR          one run
       impact_check.py --all ROOT       every */*/run.json under ROOT
"""
import json
import sys
from pathlib import Path

import pandas as pd

PRE_S = 150
P95_RISE = 0.5
ERR_RISE = 0.05


def interval_stats(spans: pd.DataFrame, t0: int, t1: int) -> dict:
    sec = spans["startTime"] // 1_000_000
    w = spans[(sec >= t0) & (sec < t1)]
    if len(w) == 0:
        return {"n": 0, "p95_us": None, "err_share": None}
    err = (w["statusCode"].fillna(0) != 0).mean()
    return {"n": int(len(w)), "p95_us": float(w["duration"].quantile(0.95)), "err_share": float(err)}


def impact_of(run_dir: Path):
    meta = json.loads((run_dir / "run.json").read_text())
    if not meta.get("inject_time"):
        return None
    t = pd.read_csv(run_dir / "traces.csv",
                    usecols=["serviceName", "operationName", "startTime", "duration", "statusCode"])
    fe = t[t["serviceName"].isin(["frontend", "frontendservice"])
           & ~t["operationName"].astype(str).str.contains("/")]
    inj, end = meta["inject_time"], meta["fault_end"]
    pre, fault = interval_stats(fe, inj - PRE_S, inj), interval_stats(fe, inj, end)
    reason = []
    if pre["n"] and not fault["n"]:
        reason.append("no_frontend_spans_during_fault")
    if pre["n"] and fault["n"]:
        if pre["p95_us"] > 0 and fault["p95_us"] >= (1 + P95_RISE) * pre["p95_us"]:
            reason.append("p95_rose_50pct")
        if fault["err_share"] - pre["err_share"] >= ERR_RISE:
            reason.append("err_share_rose_0.05")
    return {"pre": pre, "fault": fault, "impactful": bool(reason), "reason": reason}


def check(run_dir: Path) -> None:
    path = run_dir / "run.json"
    meta = json.loads(path.read_text())
    meta["impact"] = impact_of(run_dir)
    path.write_text(json.dumps(meta))
    flag = "null" if meta["impact"] is None else str(meta["impact"]["impactful"]).lower()
    print(f"{run_dir}: impact={flag}")


if __name__ == "__main__":
    if sys.argv[1] == "--all":
        for p in sorted(Path(sys.argv[2]).glob("*_*/*/run.json")):
            check(p.parent)
    else:
        check(Path(sys.argv[1]))
