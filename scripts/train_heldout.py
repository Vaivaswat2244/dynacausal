#!/usr/bin/env python3
"""Held-out-service models: retrain on D1 with one root-cause service removed.

For each of the 5 D1 root-cause services S, every TRAIN case whose root cause
is S is dropped (so validation, taken from the train cases, has none either),
then the unchanged Phase 1 training (same config, seed and split rule) is
run. The D1 test set is left whole and its result is reported split into
"S is the root cause" and "another service is".

Checkpoints: results/heldout/d1_heldout_<S>.pt, summary: results/heldout/summary.json

Usage: python scripts/train_heldout.py [configs/d1.yaml]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from src.eval import ranks_of_truth
from src.train import collate, train_once

HELD_OUT = ["checkoutservice", "currencyservice", "emailservice",
            "productcatalogservice", "recommendationservice"]


def main(cfg_path="configs/d1.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    data = torch.load(cfg["data"]["processed"], weights_only=False)
    services = data["meta"]["services"]
    out_dir = Path("results/heldout"); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for svc in HELD_OUT:
        idx = services.index(svc)
        kept = [s for s in data["samples"] if not (s["split"] == "train" and s["label"] == idx)]
        _, val_mrr, best_epoch, model = train_once({"samples": kept, "meta": data["meta"]}, cfg, quiet=True)
        torch.save(model.state_dict(), out_dir / f"d1_heldout_{svc}.pt")

        test = [s for s in data["samples"] if s["split"] == "test"]
        x, _, ei, ew, y, _ = collate(test, {"window_T": cfg["data"]["window_T"],
                                           "edge_alpha": cfg["data"]["edge_alpha"]}, services)
        model.eval()
        with torch.no_grad():
            r = ranks_of_truth(model(x, ei, ew), y)
        held = [int(v) for v, lab in zip(r, y.tolist()) if lab == idx]
        seen = [int(v) for v, lab in zip(r, y.tolist()) if lab != idx]
        summary[svc] = {
            "n_train_cases": sum(s["split"] == "train" for s in kept),
            "best_val_mrr": val_mrr, "best_epoch": best_epoch,
            "d1_test_held_out": {"n": len(held), "AC@1": sum(v == 1 for v in held) / len(held), "ranks": held},
            "d1_test_seen": {"n": len(seen), "AC@1": sum(v == 1 for v in seen) / len(seen), "ranks": seen},
        }
        print(svc, json.dumps(summary[svc]), flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
