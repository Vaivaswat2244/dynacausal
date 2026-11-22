#!/usr/bin/env python3
"""Export one recorded run from Prometheus, Jaeger and Loki into the D1 file
layout, so the Phase 1 loaders and feature extractor run on it unchanged.

Writes into <run_dir>:
  metrics.csv   1 s grid, columns {service}_{metric}   (+ "time" unix seconds)
  traces.csv    one row per span, same columns as D1
  logs.csv      time, timestamp, container_name, message, level
  inject_time.txt is already written by run_experiments.sh

Usage (with port-forwards to localhost running):
  python export_telemetry.py runs/productcatalogservice_loss/1
  python export_telemetry.py runs --all          # every run under runs/

Needs: pip install requests pandas
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import pandas as pd
import requests

PROM = "http://localhost:9090"
JAEGER = "http://localhost:16686"
LOKI = "http://localhost:3100"
NS = "boutique"
# rate() needs at least two scrapes inside its window. 30s is enough for a
# 15s scrape; run_experiments.sh raises it to 4x the scrape interval for
# coarser scrapes, otherwise the rate columns would come back empty.
RATE_WINDOW = os.environ.get("RATE_WINDOW", "30s")
SERVICES = ["adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "frontend", "loadgenerator", "paymentservice",
            "productcatalogservice", "recommendationservice", "redis-cart",
            "shippingservice"]

# Prometheus metric families per service. Names follow D1's style
# ({service}_{family}); the feature extractor maps families by name.
METRICS = {
    "container-cpu-usage-seconds-total":
        'sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="%s",pod=~"%s-.*",container!=""}[30s]))',
    "container-memory-working-set-bytes":
        'sum by (pod) (container_memory_working_set_bytes{namespace="%s",pod=~"%s-.*",container!=""})',
    "container-network-receive-bytes-total":
        'sum by (pod) (rate(container_network_receive_bytes_total{namespace="%s",pod=~"%s-.*"}[30s]))',
    "container-network-transmit-bytes-total":
        'sum by (pod) (rate(container_network_transmit_bytes_total{namespace="%s",pod=~"%s-.*"}[30s]))',
    "container-fs-writes-bytes-total":
        'sum by (pod) (rate(container_fs_writes_bytes_total{namespace="%s",pod=~"%s-.*",container!=""}[30s]))',
    "container-sockets":
        'sum by (pod) (container_sockets{namespace="%s",pod=~"%s-.*",container!=""})',
}


def prom_range(query: str, start: int, end: int, step: str = "1s") -> pd.Series:
    r = requests.get(f"{PROM}/api/v1/query_range",
                     params={"query": query, "start": start, "end": end, "step": step}, timeout=60)
    r.raise_for_status()
    res = r.json()["data"]["result"]
    if not res:
        return pd.Series(dtype=float)
    # Sum across pods of the service (usually one).
    frames = [pd.Series({int(float(t)): float(v) for t, v in m["values"]}) for m in res]
    return pd.concat(frames, axis=1).sum(axis=1)


def export_metrics(start: int, end: int) -> pd.DataFrame:
    grid = pd.Index(range(start, end + 1), name="time")
    cols = {}
    for svc in SERVICES:
        for fam, q in METRICS.items():
            s = prom_range((q % (NS, svc)).replace("[30s]", f"[{RATE_WINDOW}]"), start, end)
            cols[f"{svc}_{fam}"] = s.reindex(grid).ffill()
    df = pd.DataFrame(cols, index=grid).reset_index()
    return df


# gRPC status names -> numeric codes (D1 stores the numbers).
GRPC_CODES = {"OK": 0, "CANCELLED": 1, "UNKNOWN": 2, "INVALID_ARGUMENT": 3,
              "DEADLINE_EXCEEDED": 4, "NOT_FOUND": 5, "ALREADY_EXISTS": 6,
              "PERMISSION_DENIED": 7, "RESOURCE_EXHAUSTED": 8,
              "FAILED_PRECONDITION": 9, "ABORTED": 10, "OUT_OF_RANGE": 11,
              "UNIMPLEMENTED": 12, "INTERNAL": 13, "UNAVAILABLE": 14,
              "DATA_LOSS": 15, "UNAUTHENTICATED": 16}


def span_status(tags: dict):
    """Numeric gRPC status for a span, or None when the span reports nothing.

    Current OTel SDKs write the name under rpc.response.status_code
    ("UNAVAILABLE"); older ones wrote the number under rpc.grpc.status_code.
    Spans with only error=true / otel.status_code=ERROR (e.g. frontend HTTP
    500s) map to 2 (UNKNOWN) so they still count as errors downstream.
    """
    for key in ("rpc.grpc.status_code", "grpc.status_code", "rpc.response.status_code"):
        v = tags.get(key)
        if v is None:
            continue
        s = str(v)
        if s.lstrip("-").isdigit():
            return int(s)
        if s.upper() in GRPC_CODES:
            return GRPC_CODES[s.upper()]
    if tags.get("error") in (True, "true", "True") or tags.get("otel.status_code") == "ERROR":
        return 2
    return None


def export_traces(start: int, end: int) -> pd.DataFrame:
    rows = []
    for svc in SERVICES:
        params = {"service": svc, "start": start * 1_000_000, "end": end * 1_000_000, "limit": 200000}
        r = requests.get(f"{JAEGER}/api/traces", params=params, timeout=300)
        if r.status_code != 200:
            continue
        for tr in r.json().get("data", []):
            procs = {k: v["serviceName"] for k, v in tr["processes"].items()}
            for sp in tr["spans"]:
                tags = {t["key"]: t["value"] for t in sp.get("tags", [])}
                parent = next((rf["spanID"] for rf in sp.get("references", []) if rf["refType"] == "CHILD_OF"), "")
                code = span_status(tags)
                rows.append({
                    "time": sp["startTime"] // 1_000_000,
                    "traceID": tr["traceID"], "spanID": sp["spanID"],
                    "serviceName": procs[sp["processID"]],
                    "methodName": sp["operationName"].rsplit("/", 1)[-1],
                    "operationName": sp["operationName"],
                    "startTimeMillis": sp["startTime"] // 1000,
                    "startTime": sp["startTime"], "duration": sp["duration"],
                    "statusCode": code, "parentSpanID": parent,
                })
    df = pd.DataFrame(rows).drop_duplicates("spanID") if rows else pd.DataFrame(
        columns=["time", "traceID", "spanID", "serviceName", "methodName", "operationName",
                 "startTimeMillis", "startTime", "duration", "statusCode", "parentSpanID"])
    return df.sort_values("startTime")


def export_logs(start: int, end: int) -> pd.DataFrame:
    rows = []
    for svc in SERVICES:
        q = '{namespace="%s", pod=~"%s-.*"}' % (NS, svc)
        params = {"query": q, "start": start * 10**9, "end": end * 10**9, "limit": 5000, "direction": "forward"}
        cur = params["start"]
        while True:
            params["start"] = cur
            r = requests.get(f"{LOKI}/loki/api/v1/query_range", params=params, timeout=120)
            if r.status_code != 200:
                break
            streams = r.json()["data"]["result"]
            n = 0
            for st in streams:
                for ts, line in st["values"]:
                    n += 1
                    low = line.lower()
                    level = next((l for l in ("error", "warning", "warn", "info", "debug") if f'"{l}"' in low or f" {l} " in low or f"level={l}" in low), "")
                    rows.append({"time": int(ts) // 10**9, "timestamp": int(ts),
                                 "container_name": svc, "message": line, "level": level})
            if n < 5000 or not streams:
                break
            cur = max(int(v[0]) for st in streams for v in st["values"]) + 1
    return pd.DataFrame(rows, columns=["time", "timestamp", "container_name", "message", "level"])


def export_run(run_dir: Path) -> None:
    meta = json.loads((run_dir / "run.json").read_text())
    start, end = meta["run_start"], meta["run_end"]
    print(f"{run_dir}: {end - start}s window")
    export_metrics(start, end).to_csv(run_dir / "metrics.csv", index=False)
    tr = export_traces(start, end); tr.to_csv(run_dir / "traces.csv", index=False)
    lg = export_logs(start, end); lg.to_csv(run_dir / "logs.csv", index=False)
    print(f"  metrics ok, {len(tr)} spans, {len(lg)} log lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    root = Path(a.path)
    runs = sorted(p.parent for p in root.glob("*_*/*/run.json")) if a.all else [root]
    for r in runs:
        export_run(r)
