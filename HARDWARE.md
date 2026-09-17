# Hardware requirements

Rationale for the compute resources this project needs, with the measurements
behind each number. Written for the supervisor discussion.

## Summary

|            | Minimum    | Recommended | Driver                                         |
|------------|------------|-------------|------------------------------------------------|
| RAM        | 32 GB      | **64 GB**   | Hosting Train-Ticket's 50 services + observability |
| Storage    | 500 GB SSD | **1 TB SSD**| 110 GB of datasets + Phase-2 telemetry capture |
| CPU        | 8 cores    | **16 cores**| 50 concurrent services; parallel preprocessing |
| GPU        | none       | none        | Model is ~38k parameters; CPU training is fine |

No GPU is required. This is deliberate and worth stating up front: the model is
small (32-dim embeddings, 2 Transformer layers, 2 GAT layers, <=50 graph nodes).

## Why the current laptop cannot be used

Measured, not estimated:

```
$ df -h /
/dev/nvme0n1p4  221G  207G  4.0G  99% /
```

The single drive is 99% full with 4.0 GB free (1.3 GB of that already consumed
by this project's Python environment).

Dataset sizes were measured without downloading, by reading each archive's ZIP
central directory over HTTP range requests (`tools/probe_remote_zip.py`):

| Dataset            | Download | Extracted   | Peak need | Free   | Short by |
|--------------------|----------|-------------|-----------|--------|----------|
| D1 (RE2-OB)        | 1.19 GB  | **8.66 GB** | 9.8 GB    | 4.0 GB | ~6 GB    |
| D2 (rcabench)      | 13.4 GB  | ~97.5 GB*   | ~111 GB   | 4.0 GB | ~107 GB  |

\* extrapolated from D1's measured 7.3x expansion ratio.

**The laptop cannot hold D1 — the small dataset — let alone D2.**

Reproduce with:

    python3 tools/probe_remote_zip.py \
      "https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1"

## What the machine is for

Five workloads, of which the Kubernetes cluster is the most demanding:

1. **Dataset storage** - ~110 GB for both datasets, plus derived tensors.
2. **Preprocessing** - CPU-bound parsing of 1,430 fault cases; traces alone are
   6.37 GB in D1 and ~70 GB in D2.
3. **Training and hyperparameter search** - the paper leaves `lambda1`, `delta`,
   `m`, `edge_alpha`, `T` and `K` unspecified, so all require searching. Each
   run is cheap; there are many.
4. **Hosting the Kubernetes cluster (Phase 2)** - the system under study.
5. **Phase-2 telemetry capture** - fault-injection campaigns generate
   substantial trace volume.

### RAM breakdown for the Phase-2 cluster

| Component                              | RAM       |
|----------------------------------------|-----------|
| Train-Ticket (50 Java/Spring services) | ~24-32 GB |
| Prometheus + Jaeger + Loki             | ~8 GB     |
| k3s control plane + OS                 | ~4 GB     |
| Load generator + RCA model service     | ~3 GB     |
| **Total**                              | **~47 GB**|

Online Boutique (12 services, ~8 GB) fits comfortably in 32 GB and is the first
Phase-2 milestone. Train-Ticket is what requires 64 GB.

## Local cluster vs managed cloud (AKS/EKS)

A local cluster via **k3s** (or `kind`/MicroK8s) is preferred over managed
Kubernetes, for reasons beyond cost:

- **Cost** - Train-Ticket plus observability needs ~3-4 cloud nodes running
  continuously during multi-hour fault campaigns; roughly $300-600/month.
- **Iteration** - the experimental loop is break-measure-tweak-repeat, many
  times over. Metered infrastructure discourages exactly that.
- **Measurement validity** - one contribution of this work is reporting
  inference latency, memory and throughput, which the paper omits entirely.
  Shared cloud infrastructure has noisy neighbours and variable performance;
  a dedicated machine gives controlled, defensible timing numbers.
- **Data locality** - datasets sit on the same host as training. No egress.

**Known limitation.** A single-node cluster has near-zero inter-service network
latency and no true network partitions. Mitigations: Chaos Mesh injects network
delay, loss and partitions (already part of the fault-injection plan), and
`kind` or local VMs can provide a multi-node topology if node-level realism is
needed.

## Interim plan: Phase 1 on Colab or Kaggle

Free notebooks are enough to **start** Phase 1 on D1 now, without waiting for a
machine. The key is that preprocessing runs once: raw data is downloaded and
parsed in one session, and only the derived tensors (megabytes) are kept —
in Google Drive (Colab) or as a Kaggle Dataset / notebook output (Kaggle).
Later sessions load those tensors and never touch the raw 8.66 GB again.

|                        | Colab (free)            | Kaggle (CPU notebook)          |
|------------------------|-------------------------|--------------------------------|
| RAM                    | ~12 GB                  | ~30 GB                         |
| Scratch disk           | ~100 GB, wiped per session | tens of GB, wiped per session |
| Persistent storage     | Google Drive (15 GB free) | `/kaggle/working` output (~20 GB), private Datasets |
| Session limit          | ~12 h, idle disconnects | ~12 h                          |
| D1 (8.66 GB extracted) | Fits                    | Fits                           |

Figures are approximate and change over time; check with `!df -h` and
`!free -g` in the notebook.

What free notebooks **cannot** cover:

- **D2** — the 13.4 GB `.tar.gz` extracts to ~97.5 GB, beyond comfortable
  scratch space. Streaming the archive case by case might work but is
  unproven and fragile against session limits.
- **Phase 2** — neither platform can run Docker or a Kubernetes cluster, so the
  testbed needs a real machine regardless.

So notebooks unblock the reproduction milestone; the dedicated machine is still
required for D2 and for all of Phase 2.

## Other alternatives

1. **University HPC / research cluster** - if it offers ~250 GB persistent
   storage and long-running jobs, this covers Phase 1 fully. Phase 2 needs
   container orchestration permissions, which shared clusters often restrict.
2. **Cloud VM with a persistent disk** - workable for Phase 1 at roughly
   $20-50/month. Phase 2 costs escalate as above.

The requirement for the full project is ~250 GB persistent storage, 32-64 GB
RAM, and the ability to run multi-hour jobs and a container runtime. How that is
provided is flexible; that it is provided is not.
