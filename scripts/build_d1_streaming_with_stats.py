#!/usr/bin/env python3
"""Build d1.pt from RE2-OB.zip exactly as build_d1_streaming.py does, and
additionally save normalisation statistics fitted on the TRAIN cases.

Why: the Phase 1 pipeline normalises each case by its own min-max
(preprocess.min_max_per_series), so d1.pt stores only normalised values.
Phase 2 needs D1-train statistics to scale live-cluster features the same
way. The raw values exist only while a case is extracted, so the stats are
captured in the same streaming pass.

The sample construction is the unchanged build_sample from src/data/dataset.py,
so the resulting d1.pt matches the one trained at commit ad5a3d5.

Usage:  python scripts/build_d1_streaming_with_stats.py \
            [zip_path] [out_path] [stats_path]
"""
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data.dataset import build_sample
from src.data.loaders import OB_SERVICES, Case
from src.data.preprocess import (FEATURE_NAMES, _bucket, log_features,
                                 metric_features, trace_features)

import pandas as pd

NEEDED = {"simple_metrics.csv", "logs.csv", "traces.csv", "inject_time.txt"}
GRID_S = 15


def raw_min_max(case_dir: Path, grid_s: int = GRID_S, services=OB_SERVICES):
    """Per-(service, feature) min and max of the UN-normalised features.

    Rebuilds the grid and the raw feature tensor the same way
    preprocess.case_features does, but stops before min_max_per_series.
    """
    metrics = pd.read_csv(case_dir / "simple_metrics.csv", usecols=["time"])
    lo_t = int(_bucket(metrics["time"], grid_s).min())
    hi_t = int(_bucket(metrics["time"], grid_s).max())
    grid = np.arange(lo_t, hi_t + grid_s, grid_s)
    raw = np.concatenate([
        metric_features(case_dir, services, grid, grid_s),
        log_features(case_dir, services, grid, grid_s),
        trace_features(case_dir, services, grid, grid_s),
    ], axis=2)                                   # (N, G, D)
    return raw.min(axis=1), raw.max(axis=1)      # each (N, D)


def main(zip_path="data/RE2-OB.zip", out_path="data/processed/d1.pt",
         stats_path="results/d1_norm_stats.pt"):
    zf = zipfile.ZipFile(zip_path)
    by_case = defaultdict(list)
    for name in zf.namelist():
        parts = name.split("/")
        if len(parts) == 4 and parts[3] in NEEDED and "_" in parts[1]:
            if parts[1].rsplit("_", 1)[0] in OB_SERVICES and parts[2].isdigit():
                by_case[(parts[1], parts[2])].append(name)

    print(f"{len(by_case)} cases in archive")
    tmp_root = Path("data/raw")
    samples = []
    train_lo, train_hi = None, None
    for i, ((group, rep), members) in enumerate(sorted(by_case.items())):
        case_dir = tmp_root / "RE2-OB" / group / rep
        for m in members:
            zf.extract(m, tmp_root)
        try:
            sample = build_sample(Case.from_path(case_dir), GRID_S)
            samples.append(sample)
            if sample["split"] == "train":
                lo, hi = raw_min_max(case_dir)
                train_lo = lo if train_lo is None else np.minimum(train_lo, lo)
                train_hi = hi if train_hi is None else np.maximum(train_hi, hi)
        finally:
            shutil.rmtree(tmp_root / "RE2-OB" / group, ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(by_case)}", flush=True)

    meta = {"grid_s": GRID_S, "services": list(OB_SERVICES)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "meta": meta}, out_path)
    n_train = sum(s["split"] == "train" for s in samples)
    print(f"saved {len(samples)} samples ({n_train} train / {len(samples) - n_train} test) -> {out_path}")

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "lo": torch.tensor(train_lo, dtype=torch.float32),   # (N, D)
        "hi": torch.tensor(train_hi, dtype=torch.float32),   # (N, D)
        "services": list(OB_SERVICES),
        "feature_names": list(FEATURE_NAMES),
        "grid_s": GRID_S,
        "note": ("Elementwise min/max of raw (un-normalised) features over the "
                 "60 TRAIN cases (reps 1,2). Phase 1 training itself normalises "
                 "per case; these stats exist so live data can be scaled with "
                 "D1-train ranges without refitting."),
    }, stats_path)
    print(f"saved train normalisation stats -> {stats_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
