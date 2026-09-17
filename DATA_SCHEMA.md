# Data schema

What is actually in the datasets, established by inspecting real files. The
brief requires this before preprocessing is built, because the paper does not
describe its feature extraction.

**Status:** D1 inspected from one full fault case
(`RE2-OB/productcatalogservice_loss/1`), fetched with
`tools/fetch_zip_member.py`. Per-case layout and file sizes were confirmed for
all 90 cases from the archive index. D2 not yet inspected.

## D1 — RCAEval RE2-OB (Online Boutique)

### Layout

```
RE2-OB/{root_cause_service}_{fault_type}/{repetition}/
```

- 5 root-cause services: checkoutservice, currencyservice, emailservice,
  productcatalogservice, recommendationservice
- 6 fault types: `cpu`, `mem`, `disk`, `delay`, `loss`, `socket`
- 3 repetitions each -> **90 cases**. The label comes from the directory name.

### Timeline of a case

- 24 minutes recorded (1,441 one-second samples).
- Fault injected at the **12-minute midpoint** (`inject_time.txt`, unix
  seconds). The first half is normal, the second half faulty — so each case
  carries its own normal period, which is a natural source for `H_norm`.

### Files per case

| File | Size (avg) | Resolution | Contents |
|---|---|---|---|
| `metrics.csv` | 6.7 MB | 1 s | 421 columns named `{entity}_{metric}` |
| `simple_metrics.csv` | 1.2 MB | 1 s | 72 curated columns: cpu, mem, socket, workload, latency-50/90, diskio, error |
| `logs.csv` | 17.1 MB | per line | ~118k lines |
| `traces.csv` | 70.0 MB | per span | ~255k spans, ~17k traces |
| `tracets_lat.csv` | tiny | 15 s | latency per `{service}_{operation}` (17 columns) |
| `tracets_err.csv` | tiny | 15 s | error rate per `{service}_{operation}` |
| `logts.csv` | tiny | 15 s | count per log template (73 columns) |
| `cluster_info.json` | tiny | — | log template id -> text and container |
| `inject_time.txt` | tiny | — | fault injection time |
| `pod-node-{1,2}.csv` | tiny | — | pod -> GKE node placement |

### Services (12)

adservice, cartservice, checkoutservice, currencyservice, emailservice,
frontend, loadgenerator, paymentservice, productcatalogservice,
recommendationservice, redis, shippingservice.

**Naming mismatch:** metrics and logs say `frontend`; traces say
`frontendservice`. The loader normalises to `frontend`.

### Metrics (`metrics.csv`)

- 50 metric families per entity: container CPU/memory/filesystem/network
  counters, `container-sockets`, and Istio service-mesh metrics
  (`istio-latency-{50,90,95,99}`, `istio-request-total`, `istio-error-total`,
  `istio-bytes-*`).
- Column count per service varies (27–36), so services do not share an
  identical feature set; loadgenerator has only 8.
- Also contains per-node (`gke-...`) columns, which are not services and must
  be excluded.
- Many columns are cumulative counters (`*-total`) and need differencing into
  rates before use.
- NaN fraction: 0.2%.

### Logs (`logs.csv`)

Columns: `time, timestamp (ns), container_name, message, level, req_path, error, cluster_id, log_template`

- **Levels are inconsistent:** `info` 48,482 · `debug` 39,073 · `INFO` 16,170 ·
  missing 13,690 · `warning` 171 · `error` 15. Must be case-folded, and missing
  values handled explicitly.
- Very uneven volume: frontend 42k lines, productcatalogservice **35**.
  In this case the root cause itself is nearly silent in logs, so the log
  modality contributes almost nothing for the service that matters.
- 73 log templates, pre-clustered (`cluster_id`).

### Traces (`traces.csv`)

Columns: `time, traceID, spanID, serviceName, methodName, operationName, startTimeMillis, startTime (µs), duration (µs), statusCode, parentSpanID`

- Only **7 services emit spans**: frontend, productcatalogservice,
  currencyservice, recommendationservice, checkoutservice, emailservice,
  paymentservice. cartservice, adservice, shippingservice, redis and
  loadgenerator emit none, so they have no own trace-latency features.
- `statusCode` is a **gRPC code, not HTTP**: 0 OK (242,039), 14 UNAVAILABLE
  (198), 4 DEADLINE_EXCEEDED (12), 2 UNKNOWN (4), 13 INTERNAL (2). NaN
  (12,634) occurs only on the frontend's HTTP entry spans.
- 93.2% of spans have a parent present in the file.

### How call-graph edges are built

Edges come from **client spans**, whose `operationName` names the callee
(`hipstershop.ProductCatalogService/GetProduct`):

- `C_ij` = client calls from i to j in the window
- `R_ij` = fraction of those with a non-OK gRPC code (this is how the brief's
  "5xx and timeouts" maps onto gRPC: 4 = timeout, 14 = unavailable)

Joining server spans to parent spans was tried and rejected. It recovered only
9 edges against 14 from client spans, because services such as cartservice
emit no server spans. Under packet loss, failed calls may also produce no
server span at all, so a parent join would drop exactly the errors `R_ij`
should count.

Edges observed in the inspected case:

```
frontend -> adservice, cartservice, checkoutservice, currencyservice,
            productcatalogservice, recommendationservice, shippingservice
checkoutservice -> cartservice, currencyservice, emailservice,
                   paymentservice, productcatalogservice, shippingservice
recommendationservice -> productcatalogservice
```

### What the fault looks like (sanity check)

`productcatalogservice_loss`, 12 minutes before vs after injection:

- productcatalogservice spans: 48,937 -> 7,093 (traffic collapses)
- productcatalogservice p99 latency: 44 -> 213 µs
- most client errors appear on **frontend -> recommendationservice** (176),
  not on the root cause's own edge (15), because recommendationservice depends
  on productcatalogservice. Error counts alone would point at the wrong
  service — the propagation problem this model exists to solve.
- Edge weight frontend -> recommendationservice: 0.519 -> 0.645 (60 s windows).

### Feature dimensions (open decision 4)

Not final. The raw material per service:

| Modality | Source | Candidate width |
|---|---|---|
| Metrics | `simple_metrics.csv` or selected `metrics.csv` families | 5–8 (curated) up to 27–36 (full) |
| Logs | counts per normalised level (debug, info, warning, error, missing) | 5 |
| Traces | latency percentiles and error rate from client/server spans | ~3–4, zero for services without spans |

Choosing between `simple_metrics.csv` and the full metric set is the next
decision, and should be settled by an experiment rather than assumed.

### Open questions from inspection

1. **Resolution mismatch.** Metrics are 1 s; the pre-aggregated trace and log
   series are 15 s. Stage 1 needs one common grid; building trace and log
   features from the raw files at the chosen grid avoids depending on the
   15 s files.
2. **Services with no traces or logs.** Missing modalities must be encoded
   explicitly (zeros plus a mask), not left out, so every service keeps the
   same input width.
3. **Is loadgenerator a service?** It is counted to reach the paper's 12, but
   it is never a root cause and has almost no metrics.

## D2 — Fang et al., Train-Ticket

Not yet inspected. The archive is a single 13.4 GB `.tar.gz`, which (unlike a
ZIP) cannot be read selectively over HTTP, so inspection needs the dedicated
machine.
