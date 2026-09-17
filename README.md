# DynaCausal — clean-room reimplementation

A from-scratch reimplementation of **DynaCausal** ([arXiv:2510.22613](https://arxiv.org/abs/2510.22613)),
a root cause analysis (RCA) model for microservice systems. The authors released
no code, so this is built from the equations in the paper.

**Acceptance test:** reproduce the paper's Table 2 on dataset D1 (RCAEval RE2-OB).

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
|---|---|---|---|---|---|
| D1 — paper target | 0.769 | 0.980 | 1.000 | 0.937 | 0.873 |
| D1 — untrained model (chance) | 0.067 | 0.267 | 0.467 | 0.269 | 0.259 |

The full project brief is in [`OBJECTIVE.Md`](OBJECTIVE.Md).

## The problem

A microservice application is many small services calling each other. When one
fails, the failure spreads along those calls, and within seconds many services
look unhealthy at once. RCA is the task of ranking services by how likely each
is to be the *origin* of the incident, using their metrics, logs and traces.

## How DynaCausal works

For each time window:

1. **Temporal encoder** — a Transformer reads each service's recent
   multi-modal history *independently* and summarises it as one vector.
2. **Dynamic call graph** — a directed graph of who called whom *in this
   window*, each edge weighted by traffic volume and error rate:
   `e_ij = sigmoid(edge_alpha·Norm(C_ij) + (1−edge_alpha)·Norm(R_ij))`.
3. **Hybrid-Aware GAT** — services exchange information along the graph.
   Messages are scaled by *learned attention × observed edge weight*. This
   product is the paper's novelty and is why the layer is a custom
   `MessagePassing` subclass rather than PyG's `GATConv`.
4. **Scoring head** — an MLP turns each service's vector into a score in
   [0, 1]; the highest score is the predicted root cause.

Training combines cross-entropy with two auxiliary losses: **TCD**, a contrastive
loss pushing the root cause away from its own normal state, and **SCO**, a
ranking loss keeping the root cause above the services it affected.

## Status

| Milestone | State |
|---|---|
| 0. Environment, datasets, `DATA_SCHEMA.md` | Environment done. D1 schema documented from a real case ([`DATA_SCHEMA.md`](DATA_SCHEMA.md)); D2 needs the dedicated machine ([`HARDWARE.md`](HARDWARE.md)) |
| 1. D1 preprocessing + dynamic graph | **Call graphs built from real traces**; per-service feature extraction next |
| 2. End-to-end forward pass | **Done** |
| 3. CE training loop + eval harness | Eval harness done; training loop pending data |
| 4. TCD and SCO losses | Pending decisions on `P(r)` and `H_norm` |
| 5. Tune to the D1 row | — |
| 6. Scale to D2 | — |

53 unit tests pass.

## Repository layout

```
src/
  data/graph.py        dynamic call graph: e_ij formula, per-window graph build
  data/loaders.py      RE2 case discovery; spans -> per-window call graphs
  model/encoder.py     per-service Transformer temporal encoder
  model/hgat.py        Hybrid-Aware GAT (custom MessagePassing)
  model/head.py        per-node shared MLP scoring head
  model/dynacausal.py  full model wiring + per-window graph batching
  eval.py              AC@k, Avg@5, MRR; paper Table 2 targets
tests/                 unit tests for all of the above
tools/
  probe_remote_zip.py  measure a remote ZIP's extracted size without downloading
  fetch_zip_member.py  extract chosen files (e.g. one fault case) from a remote ZIP
DATA_SCHEMA.md         what is actually in the datasets
HARDWARE.md            compute requirements, with measurements
OBJECTIVE.Md           project brief
```

## Setup

Python 3.10+. No GPU is needed — the model has ~38k parameters.

```bash
python3 -m venv .venv
# CPU-only PyTorch; the default wheel pulls ~2.5 GB of unused CUDA libraries
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
```

### On Google Colab or Kaggle

PyTorch is preinstalled; only PyG is needed:

```python
!git clone <repo-url> dynacausal && cd dynacausal
!pip install -q torch_geometric
!python -m pytest tests/ -q
```

Colab and Kaggle disks are wiped between sessions, so preprocess once and save the derived
tensors (megabytes, not gigabytes) to Google Drive; later sessions load those
instead of re-downloading the raw dataset. Kaggle works the same way (`pip install torch_geometric`; enable Internet in notebook settings to download the dataset).

## Datasets

| | Source | Download | Extracted |
|---|---|---|---|
| **D1** RCAEval RE2-OB — 12 services, 90 cases | [Zenodo 14590730](https://zenodo.org/records/14590730) (`RE2-OB.zip`) | 1.19 GB | 8.66 GB (measured) |
| **D2** Fang et al. — Train-Ticket, 50 services, 1,430 cases | [Zenodo 17105974](https://zenodo.org/records/17105974) | 13.4 GB | ~97.5 GB (estimated) |

Each D1 case is a directory `RE2-OB/{service}_{fault}/{rep}/` holding
`metrics.csv`, `logs.csv`, `traces.csv`, `inject_time.txt`, and pre-aggregated
per-timestep series (`tracets_lat.csv`, `tracets_err.csv`, `logts.csv`). Raw
`traces.csv` is 74% of the dataset and is needed chiefly to build the per-edge
call graph, which the pre-aggregated files do not provide.

Split (from the paper): 60 train / 30 test for D1, 1,145 / 285 for D2,
stratified by fault type and injected service.

## Reimplementation decisions

The paper leaves several things unspecified. Each choice below is ours and is
recorded so results can be interpreted correctly.

| Question | Paper says | Decision |
|---|---|---|
| Is the scoring head's `W1` shared across services? | Eq. 7 writes `W1·H`, `H ∈ R^(N×D)` | **Shared, row-wise.** Consistent with the notation; lets one model run 12 or 50 services |
| How are the `T` encoder outputs pooled? | Not stated | Mean over time (configurable to last step) |
| How are ties ranked in evaluation? | Not stated | Pessimistically, so an untrained model cannot score well by accident |
| Services with no incoming edges | Not stated | Self-loop of weight 1.0, so every node has a neighbourhood |
| Min-max over identical values | Not stated | Returns zeros — no differential signal |
| `P(r)`, the "affected" services for SCO | Not defined | **Open** |
| How `H_norm` is produced for TCD | Not defined | **Open** |
| `λ1`, `δ`, `m`, `edge_alpha`, `T` | Not given | To be tuned. Given: `λ2 = 0.2`, `K = 2` GAT layers, `D_temp = 32`, 2 encoder layers |

## Findings so far

- **The edge-weight formula never gates an edge off.** Both inputs are
  normalised to [0, 1], so the sigmoid confines every `e_ij` to
  [0.500, 0.731]; an idle edge still passes about half its message.
  Implemented as written for fidelity; `compute_edge_weights(rescale=True)`
  stretches the range to [0, 1] for a later ablation.
- **Error counts alone blame the wrong service.** In
  `productcatalogservice_loss`, most client errors appear on
  frontend -> recommendationservice (176 vs 15 on the root cause's own edge),
  because recommendationservice depends on productcatalogservice.
- **Call-graph edges must come from client spans.** Some services emit no
  server spans, and failed calls under packet loss may leave none; a
  parent-span join finds 9 edges where client spans find 14.
- **Min-max normalisation is degenerate for sparse windows.** With two edges
  the normalised values are always exactly {0, 1}, whatever the traffic.

## Roadmap

**Phase 1 — reproduction (current).** Build and validate against Table 2.

**Phase 2 — extensions (after the D1 row is reproduced).**
- *Online deployment:* run the model as a streaming service against a live
  Kubernetes cluster (Prometheus / Jaeger / Loki), with fault injection via
  Chaos Mesh, and measure latency, memory and throughput — none of which the
  paper reports. `src/data/graph.py` already works on a single window with no
  stored state for this reason.
- *Label-free variant:* replace ground-truth labels with a root cause inferred
  from anomaly onset order and call-graph direction, since production systems
  have no labels.

The offline reproduction is the control for Phase 2: without it, a drop in
online accuracy cannot be told apart from an implementation bug.
