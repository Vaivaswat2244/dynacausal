#!/usr/bin/env python3
"""Hyperparameter sweep for the unspecified paper values (lambda1, delta, m,
edge_alpha). Selection is by validation MRR only; the test row of the winner
is reported at the end. Appends every run to data/processed/sweep_d1.csv.
"""
import csv
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.train import train_once

GRID = {
    "edge_alpha": [0.3, 0.5, 0.7],
    "lambda1": [0.05, 0.2],
    "delta": [0.2, 0.5],
    "m": [0.05, 0.1],
}


def main():
    cfg0 = yaml.safe_load(open("configs/d1.yaml"))
    data = torch.load(cfg0["data"]["processed"], weights_only=False)
    out = Path("data/processed/sweep_d1.csv")
    rows = []
    combos = list(itertools.product(*GRID.values()))
    # Controls: CE-only under the identical protocol, and the literal Eq. 8.
    combos += [("ce_only",), ("tcd_as_written",)]

    for k, combo in enumerate(combos):
        cfg = yaml.safe_load(open("configs/d1.yaml"))
        tag = {}
        if combo == ("ce_only",):
            cfg["loss"]["lambda1"] = 0.0
            cfg["loss"]["lambda2"] = 0.0
            tag = {"variant": "ce_only"}
        elif combo == ("tcd_as_written",):
            cfg["loss"]["tcd_as_written"] = True
            tag = {"variant": "tcd_as_written"}
        else:
            for name, val in zip(GRID, combo):
                sec = "data" if name == "edge_alpha" else "loss"
                cfg[sec][name] = val
                tag[name] = val
        t0 = time.time()
        test_m, val_mrr, best_epoch, _ = train_once(data, cfg, quiet=True)
        row = {**tag, "val_mrr": round(val_mrr, 4), "best_epoch": best_epoch,
               **{f"test_{f}": round(v, 4) for f, v in test_m.as_dict().items() if f != "n_cases"},
               "secs": int(time.time() - t0)}
        rows.append(row)
        print(f"[{k + 1}/{len(combos)}] {row}", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    best = max((r for r in rows if "variant" not in r), key=lambda r: r["val_mrr"])
    print("\nBEST BY VAL:", best)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
