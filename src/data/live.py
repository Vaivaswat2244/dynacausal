"""Phase 2: load live-cluster runs through the unchanged D1 pipeline.

A live run directory (written by phase2/export_telemetry.py) is almost a D1
case, with three differences:

- metrics.csv holds 6 Prometheus container families per service instead of
  D1's 5 curated families in simple_metrics.csv. Mapping used here:
      cpu      <- container-cpu-usage-seconds-total   (already a rate)
      mem      <- container-memory-working-set-bytes
      workload <- container-network-receive-bytes-total (request-volume proxy)
  latency-50 / latency-90 have no live source; the columns are left out and
  preprocess.metric_features zeros them, the same treatment a D1 service
  lacking a family gets.
- The redis service is named "redis-cart" live and "redis" in D1.
- Trace operation names may carry a "grpc." prefix (handled in loaders._OP_RE).

prepare_live_case() writes a derived case directory in the D1 layout; after
that Case.from_path() and build_sample() work on it unchanged. The original
exported files are never modified.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .loaders import OB_SERVICES

LIVE_FAMILY_MAP = {
    "cpu": "container-cpu-usage-seconds-total",
    "mem": "container-memory-working-set-bytes",
    "workload": "container-network-receive-bytes-total",
}
LIVE_RENAME = {"redis-cart": "redis"}


def _d1_name(svc: str) -> str:
    return LIVE_RENAME.get(svc, svc)


def prepare_live_case(run_dir: str | Path, out_root: str | Path) -> Path:
    """Write the D1-layout twin of one live run; returns the new case dir."""
    run_dir = Path(run_dir)
    out = Path(out_root) / run_dir.parent.name / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(run_dir / "metrics.csv")
    cols = {"time": m["time"]}
    for svc in OB_SERVICES:
        live_svc = next((k for k, v in LIVE_RENAME.items() if v == svc), svc)
        for fam, live_fam in LIVE_FAMILY_MAP.items():
            col = f"{live_svc}_{live_fam}"
            if col in m.columns:
                cols[f"{svc}_{fam}"] = m[col]
    pd.DataFrame(cols).to_csv(out / "simple_metrics.csv", index=False)

    logs = pd.read_csv(run_dir / "logs.csv")
    logs["container_name"] = logs["container_name"].replace(LIVE_RENAME)
    logs.to_csv(out / "logs.csv", index=False)

    traces = pd.read_csv(run_dir / "traces.csv")
    traces["serviceName"] = traces["serviceName"].replace(LIVE_RENAME)
    traces.to_csv(out / "traces.csv", index=False)

    for f in ("inject_time.txt", "run.json"):
        if (run_dir / f).exists():
            shutil.copy(run_dir / f, out / f)
    return out


def prepare_live_runs(runs_root: str | Path, out_root: str | Path) -> list[Path]:
    """Prepare every run under runs_root; returns the derived case dirs."""
    return [prepare_live_case(p.parent, out_root)
            for p in sorted(Path(runs_root).glob("*_*/*/run.json"))]
