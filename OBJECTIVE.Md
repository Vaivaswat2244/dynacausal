# DynaCausal reimplementation - build brief

## Context

Goal: a faithful, from-scratch reimplementation of DynaCausal (arXiv:2510.22613), a root cause analysis (RCA) model for microservices. There is NO public code for this paper. This is a clean-room build from the equations in the paper. The acceptance test is reproducing the numbers in the paper's Table 2 within a reasonable margin.

This brief covers Phase 1 only (reimplementation). Two research extensions ("novelties") come later and are listed at the end as Phase 2, so the architecture should leave room for them but not implement them yet.

Do not guess on the items in the "Open decisions" section. Surface them and ask.

---

## Tech stack (decided - do not re-litigate)

- Python 3.10+, PyTorch.
- PyTorch Geometric for graph plumbing, BUT the H-GAT layer needs a custom `MessagePassing` subclass, because the aggregation multiplies in a precomputed edge weight that vanilla GAT does not support (see Stage 2).
- `torch.nn.TransformerEncoder` for the temporal encoder.
- pandas / numpy / pyarrow for the time-series and trace wrangling.
- YAML configs per dataset. No hardcoded paths or dims.

---

## CRITICAL first step: get the data and inspect it before writing model code

The paper is vague on exact feature extraction, and the entire preprocessing stage depends on the real schema of each dataset. Get both datasets, inspect them, and write a short `DATA_SCHEMA.md` documenting the actual columns, sampling rates, and modality availability per service BEFORE building preprocessing. Everything downstream is shaped by what is actually in these files.

Two datasets:

- **D1** - RCAEval RE2-OB (Online Boutique). 12 services, 90 fault cases, 6 fault types, 5 fault-covered services. Split: 60 train / 30 test. This is the small, easy one. Build against this first.
- **D2** - Fang et al. 2025 (arXiv:2510.04711), Train-Ticket on Kubernetes. 50 services, 1430 fault cases, 25 fault types, 42 fault-covered services. Split: 1145 train / 285 test. This is the hard, large one. Scale to it only after D1 works.

Split rule from the paper: stratified by case type and injected service, with no fault case appearing in both train and test.

First job for Claude Code: confirm both datasets are actually downloadable and load them. RCAEval is a published benchmark with a repo. The D2 dataset is tied to arXiv:2510.04711 and needs its download location verified. If either cannot be obtained, stop and report - the project cannot proceed without data.

---

## The pipeline to build

### Stage 1 - Multi-modal alignment

Per service `i`, per timestep `t`, produce a unified feature vector by concatenating the three modalities:

- Metrics: resample to a uniform time interval. Already numeric.
- Logs: per sampling point, extract counts of events and counts per severity level. These counts are the features.
- Traces: extract latency into a per-timestep sequence.

Concatenate to `X_i^(t)` of dimension `D_metrics + D_logs + D_traces` (the three sub-dims come from the data, not the paper). Then min-max normalize each feature to [0,1]. Then segment into windows of length `T`. Model input per service is a `T`-length sequence of these vectors.

**Dynamic call graph.** For every window `t`, build a weighted directed graph. Edge from calling service `i` to called service `j`:

```
e_ij^(t) = sigmoid( edge_alpha * Norm(C_ij^(t)) + (1 - edge_alpha) * Norm(R_ij^(t)) )
```

`C_ij` = request count on the i->j call in that window. `R_ij` = status-code error rate (5xx, timeouts) on that call. `edge_alpha` in [0,1] is a hyperparameter. Direction matters: `e_ij != e_ji`. This dynamic graph is the single most important component per the ablation, so get it right and unit-test it.

### Stage 2 - Representation learning

**Temporal encoder (per service, independent).** Standard Transformer encoder over the `T` time steps of `X_i`. Q/K/V projections, multi-head self-attention across time, feed-forward with residual + layernorm. Output one embedding per service:

```
H_i^(t) in R^(D_temp)
```

Use `D_temp = 32` and `2` Transformer layers. These are the paper's best-found settings; deeper overfits. Make both config values, not literals.

**Hybrid-Aware GAT (H-GAT).** Dual mechanism. First, the standard GAT attention coefficient between service `i` and neighbour `j`:

```
attn_ij^(k) = softmax_over_N(i)( LeakyReLU( a^T [ W H_i^(k-1) || W H_j^(k-1) ] ) )
```

Then aggregate, multiplying in BOTH the learned attention AND the Stage-1 edge weight `e_ij`:

```
H_i^(k) = sigmoid( sum_{j in N(i)}  attn_ij^(k) * e_ij^(t) * W^(k) H_j^(k-1) )
```

The `e_ij` factor is what makes this non-standard and forces the custom `MessagePassing` subclass. Output after the final layer `K`: `H^(s,t)` of shape `(N, D_spat)`, one row per service.

Naming note: the paper reuses the symbol alpha for both the edge-weight blend coefficient and the GAT attention coefficient. Name them `edge_alpha` and `attn` in code so they never collide.

**Scoring head.** Two-layer MLP mapping each service's embedding to a score in [0,1]:

```
S^(t) = sigmoid( W2 * ReLU( W1 * H^(s,t) + b1 ) + b2 )
```

Rank services by `S_i` descending; top one is the predicted root cause. See Open decisions for the W1 shape question, which determines whether the model is locked to a fixed service count.

### Stage 3 - Losses (training only)

Three terms.

**Cross-entropy** over services: treat each case as V-way classification, true class = the labelled root-cause service.

**TCD (temporal causal disentanglement)** - contrastive hinge on cosine similarities:

```
L_TCD = sum_{i != r}  max(0, delta - cos(H_r^anom, H_r^norm) + cos(H_i^anom, H_i^norm))
```

`H_i^anom` is the current embedding, `H_i^norm` is the same service's normal-period embedding, `r` is the true root cause, `delta` is a margin. Intuition: push the root cause's current state far from its own normal; keep everyone else close to theirs. See Open decisions for how `H_norm` is obtained.

**SCO (spatial causal ordering)** - pairwise ranking hinge:

```
L_rank = sum_{i in P(r)}  max(0, m - (S_r - S_i))
```

Forces the root cause's score above every affected service's by margin `m`. `P(r)` is the set of affected upstream/downstream services. See Open decisions - the paper never defines how `P(r)` is built, and SCO cannot be implemented until this is settled.

**Total:**

```
L_total = L_CE + lambda1 * L_TCD + lambda2 * L_rank
```

Known hyperparameter: `lambda2 = 0.2`. `lambda1`, `delta`, `m`, `edge_alpha`, `T`, and number of GAT layers `K` are unspecified and need tuning.

---

## Evaluation

Metrics, computed on the test split:

- `AC@k` = fraction of cases where the true root cause is in the top `k`. Report k = 1, 3, 5.
- `Avg@5` = (1/5) * sum_{k=1..5} AC@k. (This is the paper's own definition; implement it as written even though it is unusual.)
- `MRR` = (1/N) * sum_i 1 / rank(true root cause in case i).

### Reproduction targets (paper Table 2 - this is the acceptance test)

| Dataset | AC@1 | AC@3 | AC@5 | Avg@5 | MRR |
|---|---|---|---|---|---|
| D1 | 0.769 | 0.980 | 1.000 | 0.937 | 0.873 |
| D2 | 0.481 | 0.696 | 0.819 | 0.680 | 0.626 |

Getting close to the D1 row is the milestone that says the implementation is fundamentally correct. D2 is harder and more sensitive to preprocessing.

---

## Open decisions (do NOT guess - surface these to me)

1. **`P(r)` construction.** The paper says `P(r)` is "affected downstream and upstream services" but never defines the rule. Blocks SCO entirely. Candidate definitions to discuss: all services on the call path between the root cause and observed anomalies; all k-hop neighbours in the call graph; all services whose anomaly score exceeds a threshold. Pick one, document it, treat it as a design choice of the reimplementation.

2. **`H_norm` sourcing.** How is each service's "normal state representation" produced? Likely: run the same temporal encoder on a normal-period window for that service. Open question: a single fixed reference normal window per service, or a rolling recent-normal window? Decide and document.

3. **MLP `W1` shape.** Is `W1` applied per-node (shared weights across services, so the model is topology-agnostic and the same code handles D1's 12 services and D2's 50), or applied to the flattened `(N, D)` matrix (locking the model to a fixed `N`)? Strong recommendation: per-node/shared, so one implementation runs both datasets and Phase 2 transfer experiments stay possible. Flag if the paper's phrasing forces otherwise.

4. **Feature sub-dimensions.** `D_metrics`, `D_logs`, `D_traces` come from the actual data. Fill in after data inspection, per dataset.

5. **Anomaly/normal window pairing** for TCD depends on decision 2. Resolve together.

---

## Suggested repo structure

```
dynacausal/
  data/                 # download scripts + raw (gitignored)
  src/
    data/
      loaders.py        # per-dataset loaders (D1, D2)
      preprocess.py     # modality -> MTS, normalize, window
      graph.py          # dynamic call graph (edge weights)
    model/
      encoder.py        # transformer temporal encoder
      hgat.py           # hybrid-aware GAT (custom MessagePassing)
      head.py           # MLP scoring head
      dynacausal.py     # full model wiring
    losses.py           # CE + TCD + SCO
    train.py
    eval.py             # AC@k, Avg@5, MRR
  configs/
    d1.yaml
    d2.yaml
  tests/
  DATA_SCHEMA.md        # written after data inspection
  README.md
```

---

## Build order (milestones)

0. Environment set up. Both datasets downloaded and loading. `DATA_SCHEMA.md` written.
1. D1 preprocessing: produce `X_i^(t)` tensors and the dynamic graph. Unit-test the edge-weight computation.
2. Model forward pass end to end (encoder -> H-GAT -> head) producing a score vector on random init. Shapes correct, no training yet.
3. CE-only training loop plus the eval harness. Get a non-trivial AC@1 on D1. This proves the plumbing.
4. Add TCD, then SCO. Resolve decisions 1 and 2 first.
5. Full-loss training on D1. Tune to approach the D1 target row.
6. Scale to D2. Re-tune as needed.
7. Stop. Phase 2 begins only after the D1 row is reproduced.

---

## Phase 2 hooks (NOT now - for architecture awareness only)

Two extensions will be built on top of the working reimplementation. Keep the code modular enough that these attach cleanly; do not implement them.

- **Label-free variant (ML track).** Replaces the labels that TCD and SCO consume. An autoencoder trained on normal windows produces reconstruction residuals; change-point detection on the residuals gives a per-service anomaly onset time; onset order plus call-graph direction yields a pseudo root cause and pseudo `P(r)` that feed the existing losses. Implication for Phase 1: keep the loss module's dependence on `r` and `P(r)` behind a clean interface, so the label source can be swapped for a computed one without touching the model.

- **Online deployment (systems track).** Wrap the trained model as a streaming service ingesting live from Prometheus / Jaeger / Loki, building the dynamic graph per window in real time, and emitting rankings under a latency budget. Also measure inference latency, memory, and throughput (the paper reports none). Implication for Phase 1: keep preprocessing and graph construction callable on a single live window, not only on a whole recorded dataset.
