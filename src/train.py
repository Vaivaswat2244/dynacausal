"""Full-loss training on D1 (milestones 4-5). Usage:

    python -m src.train configs/d1.yaml [--override key=value ...]

Dataset samples hold full feature series + per-bucket call stats; windows and
graphs are cut here (window_sample), so window_T / edge_alpha / offset are
pure config knobs.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml

from .data.dataset import window_sample
from .eval import PAPER_TARGETS, evaluate
from .losses import neighbour_mask, total_loss
from .model.dynacausal import DynaCausal, batch_graphs


def collate(samples, cfg, services):
    """Cut windows + graphs for a list of samples and batch them."""
    xs, xns, eis, ews = [], [], [], []
    for s in samples:
        x, xn, ei, ew = window_sample(s, cfg["window_T"], cfg["edge_alpha"])
        xs.append(x); xns.append(xn); eis.append(ei); ews.append(ew)
    n = len(services)
    ei_b, ew_b = batch_graphs(eis, ews, n)
    y = torch.tensor([s["label"] for s in samples])
    # per-sample (un-offset) edges for P(r), padded to equal width
    e_max = max(e.size(1) for e in eis)
    ei_pad = torch.full((len(eis), 2, e_max), -1, dtype=torch.long)
    for k, e in enumerate(eis):
        ei_pad[k, :, :e.size(1)] = e
    p_mask = neighbour_mask(ei_pad, y, n)
    return torch.stack(xs), torch.stack(xns), ei_b, ew_b, y, p_mask


def split_train_val(train_samples):
    """Hold out repetition 2 of every 3rd (service, fault) group as validation.

    Deterministic and stratified-ish: 10 of the 60 training cases, never seen
    by the optimiser, used solely for early stopping / model selection.
    """
    groups = sorted({s["case"].split("/")[0] for s in train_samples})
    val_groups = set(groups[::3])
    val = [s for s in train_samples
           if s["case"].split("/")[0] in val_groups and s["case"].endswith("/2")]
    val_ids = {s["case"] for s in val}
    return [s for s in train_samples if s["case"] not in val_ids], val


def run_eval(model, batch):
    model.eval()
    x, xn, ei, ew, y, _ = batch
    with torch.no_grad():
        return evaluate(model(x, ei, ew), y)


def train_once(data, cfg, quiet=False):
    """One full training run. Returns (test_metrics, val_metrics, best_epoch)."""
    d, m, t, l = cfg["data"], cfg["model"], cfg["train"], cfg["loss"]
    torch.manual_seed(t["seed"])
    services = data["meta"]["services"]
    n = len(services)

    train_all = [s for s in data["samples"] if s["split"] == "train"]
    test = [s for s in data["samples"] if s["split"] == "test"]
    train, val = split_train_val(train_all)
    if not quiet:
        print(f"{len(train)} train / {len(val)} val / {len(test)} test")

    wcfg = {"window_T": d["window_T"], "edge_alpha": d["edge_alpha"]}
    val_batch = collate(val, wcfg, services)
    test_batch = collate(test, wcfg, services)

    model = DynaCausal(**m)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    best = {"mrr": -1.0, "epoch": 0, "state": None}
    since_best = 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(train))
        for i in range(0, len(train), t["batch_size"]):
            batch = [train[j] for j in perm[i:i + t["batch_size"]]]
            x, xn, ei, ew, y, p_mask = collate(batch, wcfg, services)
            logits, h_temp, _ = model(x, ei, ew, return_logits=True, return_embeddings=True)
            scores = torch.sigmoid(logits)
            h_norm = model.encode(xn)        # same encoder, normal window
            loss, parts = total_loss(logits, scores, h_temp, h_norm, y, p_mask,
                                     l["lambda1"], l["lambda2"], l["delta"], l["m"],
                                     l.get("tcd_as_written", False))
            opt.zero_grad(); loss.backward(); opt.step()

        if epoch % t["eval_every"] == 0:
            vm = run_eval(model, val_batch)
            if vm.mrr > best["mrr"]:
                best = {"mrr": vm.mrr, "epoch": epoch,
                        "state": copy.deepcopy(model.state_dict())}
                since_best = 0
            else:
                since_best += 1
            if not quiet and epoch % (t["eval_every"] * 5) == 0:
                print(f"epoch {epoch:4d}  loss {loss.item():.3f} "
                      f"(ce {parts['ce']:.3f} tcd {parts['tcd']:.3f} sco {parts['sco']:.3f})  "
                      f"val MRR {vm.mrr:.3f}  best {best['mrr']:.3f}@{best['epoch']}")
            if since_best >= t["patience"]:
                break

    model.load_state_dict(best["state"])
    return run_eval(model, test_batch), best["mrr"], best["epoch"], model


def main(argv):
    cfg = yaml.safe_load(open(argv[0] if argv else "configs/d1.yaml"))
    for ov in argv[1:]:
        if "=" in ov:
            key, val = ov.split("=", 1)
            sec, name = key.split(".", 1)
            cfg[sec][name] = yaml.safe_load(val)
    data = torch.load(cfg["data"]["processed"], weights_only=False)
    test_m, val_mrr, best_epoch, model = train_once(data, cfg)
    print(f"\nbest val MRR {val_mrr:.3f} at epoch {best_epoch}")
    print("RESULT test :", test_m)
    print("paper D1    :", PAPER_TARGETS["d1"])
    out = Path(cfg["data"]["processed"]).with_name("d1_full_model.pt")
    torch.save(model.state_dict(), out)
    print("checkpoint  :", out)


if __name__ == "__main__":
    main(sys.argv[1:])
