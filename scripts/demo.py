#!/usr/bin/env python3
"""End-to-end demonstration of the current state of the project.

Run:  .venv/bin/python scripts/demo.py

Walks through the four things that are working: the real data, the problem the
model has to solve, the dynamic call graph built from real traces, and the
model's forward pass. Needs one fault case in data/raw (see README).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # keep the demo output readable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import src/ when run directly

import pandas as pd
import torch

from src.data.loaders import Case, case_graphs, load_client_calls
from src.eval import PAPER_TARGETS, evaluate
from src.model.dynacausal import DynaCausal, batch_graphs

CASE = Path("data/raw/RE2-OB/productcatalogservice_loss/1")


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    if not CASE.exists():
        raise SystemExit(
            f"{CASE} not found. Fetch it with:\n"
            "  .venv/bin/python tools/fetch_zip_member.py \\\n"
            '    "https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1" \\\n'
            f'    "{CASE.parent.parent.name}/{CASE.parent.name}/{CASE.name}/" data/raw'
        )

    case = Case.from_path(CASE)

    rule("1. THE DATA  (one real fault case from the RCAEval benchmark)")
    metrics = pd.read_csv(CASE / "metrics.csv")
    traces = pd.read_csv(CASE / "traces.csv")
    logs = pd.read_csv(CASE / "logs.csv")
    minutes = (metrics.time.max() - metrics.time.min()) / 60
    print(f"  fault injected into : {case.root_cause}  (type: {case.fault})")
    print(f"  recording length    : {minutes:.0f} minutes, 1 sample/second")
    print(f"  fault starts at     : minute {(case.inject_time - metrics.time.min()) / 60:.0f}"
          f"  -> first half normal, second half faulty")
    print(f"  metrics             : {metrics.shape[1] - 1} columns")
    print(f"  log lines           : {len(logs):,}")
    print(f"  trace records       : {len(traces):,}")
    print("\n  Problems found in the data (documented in DATA_SCHEMA.md):")
    print(f"    - log levels are inconsistent: "
          f"{', '.join(f'{k}={v}' for k, v in logs.level.value_counts().head(4).items())}")
    print(f"    - the faulty service writes only {(logs.container_name == case.root_cause).sum()} "
          f"log lines, vs {logs.container_name.value_counts().iloc[0]:,} for the busiest")
    print(f"    - only {traces.serviceName.nunique()} of 12 services produce traces at all")

    rule("2. THE PROBLEM  (why counting errors is not enough)")
    calls = load_client_calls(traces)
    errs = calls[calls.error].groupby(["caller", "callee"]).size().sort_values(ascending=False)
    print("  Failed calls, by which pair of services they happened between:")
    for (a, b), n in errs.head(4).items():
        flag = "  <-- the actual culprit" if b == case.root_cause else ""
        print(f"    {a:22} -> {b:22} {n:5}{flag}")
    print(f"\n  The fault was in {case.root_cause}, but most failures show up")
    print("  elsewhere, because those services depend on it. Blaming whoever has")
    print("  the most errors gives the wrong answer. This is what the model fixes.")

    rule("3. OUR CALL GRAPH  (built from the real traces)")
    graphs = case_graphs(case, window_s=60)
    print(f"  Built {len(graphs)} graphs, one per minute. Each connection is scored")
    print("  by how much traffic it carried and how much of it failed.\n")
    watch = [("frontend", "recommendationservice"), ("frontend", "productcatalogservice")]
    before = {w: None for w in watch}
    after = {w: None for w in watch}
    for start, g in graphs:
        rel = (start - case.inject_time) // 60
        if rel not in (-2, 1):
            continue
        for k, (i, j) in enumerate(g.edge_index.t().tolist()):
            pair = (g.services[i], g.services[j])
            if pair in watch:
                (before if rel == -2 else after)[pair] = g.edge_weight[k].item()
    print(f"  {'connection':<48}{'before':>9}{'after':>9}")
    for pair in watch:
        b, a = before[pair], after[pair]
        mark = "   <-- reacts to the fault" if a - b > 0.05 else ""
        print(f"  {pair[0] + ' -> ' + pair[1]:<48}{b:>9.3f}{a:>9.3f}{mark}")
    print("\n  The graph changes when the fault starts. That is the paper's main idea,")
    print("  and it is working on real data.")

    rule("4. THE MODEL  (runs end to end; not trained yet)")
    model = DynaCausal(d_in=17).eval()
    n = 12
    x = torch.randn(2, n, 30, 17)
    ei, ew = batch_graphs([graphs[0][1].edge_index] * 2, [graphs[0][1].edge_weight] * 2, n)
    scores = model(x, ei, ew)
    print(f"  input  : {tuple(x.shape)}  (2 cases, 12 services, 30 time steps, 17 features)")
    print(f"  output : {tuple(scores.shape)}  one suspicion score per service")
    print(f"  size   : {sum(p.numel() for p in model.parameters()):,} parameters (small; no GPU needed)")

    gen = torch.Generator().manual_seed(0)
    untrained = evaluate(torch.rand(90, 12, generator=gen),
                         torch.randint(0, 12, (90,), generator=gen))
    print("\n  Scoreboard (how often the true culprit is ranked first, and overall):")
    print(f"    untrained model, random guessing : {untrained}")
    print(f"    the paper's published result     : {PAPER_TARGETS['d1']}")
    print("\n  Closing that gap is the goal of the next stage.")
    print()


if __name__ == "__main__":
    main()
