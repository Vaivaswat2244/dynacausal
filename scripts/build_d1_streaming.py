#!/usr/bin/env python3
"""Build the D1 dataset directly from RE2-OB.zip, one case at a time.

Written for machines that cannot hold the 8.66 GB extracted dataset. Peak disk
use is the zip plus one case (~100 MB): each case is extracted, preprocessed
into tensors, and deleted before the next one starts.

Usage:  python scripts/build_d1_streaming.py [zip_path] [out_path]
"""
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data.dataset import build_sample
from src.data.loaders import OB_SERVICES, Case

# The only files preprocessing reads. metrics.csv (421 columns) is unused by
# the current feature recipe, and skipping it saves extraction time and disk.
NEEDED = {"simple_metrics.csv", "logs.csv", "traces.csv", "inject_time.txt"}

GRID_S, WINDOW_T, EDGE_ALPHA = 15, 20, 0.5


def main(zip_path="data/RE2-OB.zip", out_path="data/processed/d1.pt"):
    zf = zipfile.ZipFile(zip_path)
    by_case = defaultdict(list)
    for name in zf.namelist():
        parts = name.split("/")
        # RE2-OB/{service}_{fault}/{rep}/{file}
        if len(parts) == 4 and parts[3] in NEEDED and "_" in parts[1]:
            if parts[1].rsplit("_", 1)[0] in OB_SERVICES:
                by_case[(parts[1], parts[2])].append(name)

    print(f"{len(by_case)} cases in archive")
    tmp_root = Path("data/raw")
    samples = []
    for i, ((group, rep), members) in enumerate(sorted(by_case.items())):
        case_dir = tmp_root / "RE2-OB" / group / rep
        for m in members:
            zf.extract(m, tmp_root)
        try:
            samples.append(build_sample(Case.from_path(case_dir),
                                        GRID_S, WINDOW_T, EDGE_ALPHA))
        finally:
            shutil.rmtree(tmp_root / "RE2-OB" / group, ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(by_case)}", flush=True)

    meta = {"grid_s": GRID_S, "window_T": WINDOW_T, "edge_alpha": EDGE_ALPHA,
            "services": list(OB_SERVICES)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "meta": meta}, out_path)
    n_train = sum(s["split"] == "train" for s in samples)
    print(f"saved {len(samples)} samples ({n_train} train / {len(samples) - n_train} test) -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
