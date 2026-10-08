"""CE-only training on D1 (milestone 3). Usage:

    python -m src.train configs/d1.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from .data.dataset import build_dataset
from .eval import PAPER_TARGETS, evaluate
from .model.dynacausal import DynaCausal, batch_graphs


def collate(samples, n_services):
    x = torch.stack([s["x"] for s in samples])
    ei, ew = batch_graphs([s["edge_index"] for s in samples],
                          [s["edge_weight"] for s in samples], n_services)
    y = torch.tensor([s["label"] for s in samples])
    return x, ei, ew, y


def run_eval(model, samples, n):
    model.eval()
    with torch.no_grad():
        x, ei, ew, y = collate(samples, n)
        return evaluate(model(x, ei, ew), y)


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    torch.manual_seed(t["seed"])

    proc = Path(d["processed"])
    if proc.exists():
        data = torch.load(proc, weights_only=False)
    else:
        print("building dataset (one-time)...")
        data = build_dataset(d["raw_root"], d["processed"], d["grid_s"],
                             d["window_T"], d["edge_alpha"])
    samples, services = data["samples"], data["meta"]["services"]
    n = len(services)
    train = [s for s in samples if s["split"] == "train"]
    test = [s for s in samples if s["split"] == "test"]
    print(f"{len(train)} train / {len(test)} test / {n} services / d_in={m['d_in']}")

    model = DynaCausal(**m)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(train))
        total = 0.0
        for i in range(0, len(train), t["batch_size"]):
            batch = [train[j] for j in perm[i:i + t["batch_size"]]]
            x, ei, ew, y = collate(batch, n)
            logits = model(x, ei, ew, return_logits=True)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(batch)
        if epoch % t["eval_every"] == 0 or epoch == t["epochs"]:
            tr = run_eval(model, train, n)
            te = run_eval(model, test, n)
            print(f"epoch {epoch:4d}  loss {total / len(train):.4f}  "
                  f"train AC@1 {tr.ac1:.3f}  | test {te}")

    print("\nfinal test :", run_eval(model, test, n))
    print("paper D1   :", PAPER_TARGETS["d1"])
    ckpt = proc.with_name(proc.stem + "_ce_model.pt")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print("checkpoint :", ckpt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/d1.yaml")
