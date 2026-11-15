# Task for Claude Code: evaluate the Phase 1 model on live cluster runs

Run from the repo root (the dynacausal repo). Read OBJECTIVE.Md,
DATA_SCHEMA.md and PROJECT_CONTEXT.md first. Rules from the Phase 1 task
still apply: plain-English explanations, never adjust or round results, never
backdate commits, keep tests passing, no em dashes.

## Input

`data/live/runs/{service}_{fault}/{rep}/` holds runs recorded on our k3s
cluster, in the D1 layout: metrics.csv (1 s grid, `{service}_{family}`
columns), traces.csv (D1 columns), logs.csv, inject_time.txt and run.json
(run_start, inject_time, fault_end, run_end). `none_none/*` runs have no
fault. Fault start is at a RANDOM offset inside each run.

## Step 0: find the trained model

Look in `results/` and any checkpoint path the Phase 1 training code uses.
We need the full-loss model (CE + TCD + SCO) that produced the reported D1
numbers (AC@1 0.833). If no checkpoint was saved:
1. D1 must be re-downloaded (data/raw was deleted). Use the same command as
   the Phase 1 task and the same split and config as the reported run.
2. Retrain and SAVE the checkpoint plus the fitted normalisation statistics
   this time. Report the retrained D1 test numbers alongside the old ones.

## Step 1: load live runs through the existing pipeline

Extend `src/data/loaders.py` / `features.py` only where needed so a live run
loads like a D1 case. Expected differences to handle, and report which you
hit:
- Service naming: `redis-cart` vs D1's `redis`; traces may say `frontend`
  rather than `frontendservice`.
- Metric column names: live metrics.csv has 6 families per service with D1
  style names. Map them onto the families the feature extractor expects;
  fill families that do not exist live with zeros and set the presence mask.
- Trace `statusCode` may be missing or a string; treat non-zero as error.
- Log levels extracted loosely; case-fold as for D1.
Apply the normalisation statistics FITTED ON D1 TRAIN; do not refit on live
data.

## Step 2: evaluate without the fault time

For each run, slide windows of the Phase 1 window length over the whole run
(stride = grid step, e.g. 15 s). For every window record the model's score
vector and the top-ranked service. Save per run to
`results/live/{service}_{fault}_{rep}.json`.

Then compute, using run.json for the ground truth:
1. **Accuracy at fault time** (comparable to D1): the window starting at
   inject_time. AC@1, AC@3, AC@5, MRR over all fault runs.
2. **Accuracy without fault time**: for each run, the window with the
   highest top score across the run, i.e. the model's own choice of when the
   fault is. Same metrics.
3. **Time to diagnosis**: first window start, after inject_time, where the
   top-1 service equals the true service, minus inject_time. Report median
   and the share of runs never diagnosed within the fault duration.
4. **False alarms**: on `none_none` runs and on the pre-fault part of fault
   runs, the share of windows whose top score exceeds a threshold. Report
   the curve for thresholds 0.5, 0.7, 0.9, plus the same thresholds applied
   to the post-fault windows, so the trade-off is visible.
5. **Per fault type** breakdown of 1 and 2.
6. **Pipeline latency and memory**: time per window for loading features,
   building the graph and the forward pass, and peak RSS.

Also run the two simple baselines on the same runs for comparison:
- metric-jump: service with the largest normalised change in its own
  metrics between the window and the preceding window;
- error-count: service with the most failed calls on its own edges.

## Step 3: report

Give the user, in plain text: the number of runs per service and fault type;
every number from Step 2 in tables; the baselines next to the model; the
D1 test numbers of the model used; the data differences handled in Step 1;
anything that looked wrong in the live data (empty traces, services with no
metrics). Do not interpret the numbers beyond stating them.
