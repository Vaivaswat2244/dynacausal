"""Stage 1: raw case files -> per-service feature tensors on a shared time grid.

Design decisions (the paper specifies none of this; see DATA_SCHEMA.md):

- Grid: 15 s buckets. Metrics (1 s) are averaged into buckets; logs and traces
  are counted/aggregated per bucket. Matches the resolution of RCAEval's own
  derived files (tracets_*, logts).
- Metrics: the 5 curated families below from simple_metrics.csv. Full
  metrics.csv (421 columns) is a later experiment, not the default.
- Logs: line counts per severity level, case-folded; missing level -> "other".
- Traces: per service span count, mean duration, error rate (gRPC code != 0).
  Server spans attribute load to the service doing the work.
- Normalisation: min-max per (service, feature) over the WHOLE recording.
  The first half of every case is fault-free, so a post-normalisation value
  near 1 means "high for this service", which is the deviation signal the
  model needs. Constant features normalise to 0 (same rule as graph.py).
- Missing modalities (5 services emit no traces) become all-zero features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .loaders import _TRACE_NAME_FIX, OB_SERVICES, Case

METRIC_FAMILIES = ["cpu", "mem", "latency-50", "latency-90", "workload"]
LOG_LEVELS = ["info", "debug", "warning", "error", "other"]
TRACE_FEATS = ["span_count", "span_mean_dur", "span_err_rate"]

FEATURE_NAMES = (
    [f"metric_{f}" for f in METRIC_FAMILIES]
    + [f"log_{l}" for l in LOG_LEVELS]
    + [f"trace_{t}" for t in TRACE_FEATS]
)
D_IN = len(FEATURE_NAMES)  # 13


def _bucket(sec: pd.Series, grid_s: int) -> pd.Series:
    return (sec // grid_s) * grid_s


def metric_features(case_dir, services, grid, grid_s) -> np.ndarray:
    """(N, G, len(METRIC_FAMILIES)) from simple_metrics.csv, averaged per bucket."""
    df = pd.read_csv(case_dir / "simple_metrics.csv")
    df["t"] = _bucket(df["time"], grid_s)
    agg = df.groupby("t").mean(numeric_only=True)
    out = np.zeros((len(services), len(grid), len(METRIC_FAMILIES)))
    pos = {t: i for i, t in enumerate(grid)}
    rows = [pos[t] for t in agg.index if t in pos]
    kept = [t for t in agg.index if t in pos]
    for si, svc in enumerate(services):
        for fi, fam in enumerate(METRIC_FAMILIES):
            col = f"{svc}_{fam}"
            if col in agg.columns:
                out[si, rows, fi] = agg.loc[kept, col].to_numpy()
    return np.nan_to_num(out)


def log_features(case_dir, services, grid, grid_s) -> np.ndarray:
    """(N, G, 5) line counts per folded severity level."""
    df = pd.read_csv(case_dir / "logs.csv", usecols=["timestamp", "container_name", "level"])
    df["t"] = _bucket(df["timestamp"] // 1_000_000_000, grid_s)   # ns -> s
    lvl = df["level"].astype(str).str.lower()
    df["lvl"] = np.where(lvl.isin(LOG_LEVELS[:4]), lvl, "other")  # NaN/unknown -> other
    counts = df.groupby(["container_name", "t", "lvl"]).size()
    out = np.zeros((len(services), len(grid), len(LOG_LEVELS)))
    pos = {t: i for i, t in enumerate(grid)}
    sidx = {s: i for i, s in enumerate(services)}
    lidx = {l: i for i, l in enumerate(LOG_LEVELS)}
    for (svc, t, lvl_), n in counts.items():
        if svc in sidx and t in pos:
            out[sidx[svc], pos[t], lidx[lvl_]] = n
    return out


def trace_features(case_dir, services, grid, grid_s) -> np.ndarray:
    """(N, G, 3) per-service span count, mean duration, error rate."""
    df = pd.read_csv(case_dir / "traces.csv",
                     usecols=["serviceName", "startTime", "duration", "statusCode"])
    df["svc"] = df["serviceName"].replace(_TRACE_NAME_FIX)
    df["t"] = _bucket(df["startTime"] // 1_000_000, grid_s)       # us -> s
    df["err"] = (df["statusCode"].fillna(0) != 0).astype(float)
    g = df.groupby(["svc", "t"]).agg(n=("duration", "size"),
                                     dur=("duration", "mean"),
                                     err=("err", "mean"))
    out = np.zeros((len(services), len(grid), len(TRACE_FEATS)))
    pos = {t: i for i, t in enumerate(grid)}
    sidx = {s: i for i, s in enumerate(services)}
    for (svc, t), row in g.iterrows():
        if svc in sidx and t in pos:
            out[sidx[svc], pos[t]] = [row["n"], row["dur"], row["err"]]
    return out


def min_max_per_series(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max each (service, feature) series over time to [0,1]; constant -> 0."""
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    rng = hi - lo
    out = np.where(rng > eps, (x - lo) / np.where(rng > eps, rng, 1.0), 0.0)
    return out


def case_features(case: Case, grid_s: int = 15, services=OB_SERVICES):
    """Full-recording feature tensor for one case.

    Returns:
        grid:  (G,) bucket start times (unix s)
        feats: (N, G, D_IN) normalised to [0,1]
    """
    metrics = pd.read_csv(case.path / "simple_metrics.csv", usecols=["time"])
    lo = int(_bucket(metrics["time"], grid_s).min())
    hi = int(_bucket(metrics["time"], grid_s).max())
    grid = np.arange(lo, hi + grid_s, grid_s)

    feats = np.concatenate([
        metric_features(case.path, services, grid, grid_s),
        log_features(case.path, services, grid, grid_s),
        trace_features(case.path, services, grid, grid_s),
    ], axis=2)
    return grid, min_max_per_series(feats)
