"""Loaders for RCAEval RE2 fault cases (dataset D1, Online Boutique).

Layout, per case:  RE2-OB/{root_cause_service}_{fault_type}/{repetition}/
See DATA_SCHEMA.md for the columns in each file.

Only the pieces needed to build per-window call graphs live here for now;
feature extraction for the temporal encoder comes next.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .graph import CallGraph, build_call_graph

# The 12 Online Boutique services. RCAEval's metrics use "frontend" while the
# traces say "frontendservice"; everything is normalised to the metric names.
OB_SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice",
    "emailservice", "frontend", "loadgenerator", "paymentservice",
    "productcatalogservice", "recommendationservice", "redis",
    "shippingservice",
]
_TRACE_NAME_FIX = {"frontendservice": "frontend"}

# gRPC status codes. 0 is OK. The brief's "5xx and timeouts" corresponds to any
# non-OK code here, which includes 4 (DEADLINE_EXCEEDED, i.e. timeout) and
# 14 (UNAVAILABLE). HTTP spans from the frontend carry NaN and count as OK.
GRPC_OK = 0

# Client spans name the *called* service in their operation, e.g.
# "hipstershop.ProductCatalogService/GetProduct" or with a leading slash.
_OP_RE = re.compile(r"^/?hipstershop\.(\w+)/")


def callee_from_operation(operation: str) -> str | None:
    """'hipstershop.ProductCatalogService/GetProduct' -> 'productcatalogservice'."""
    m = _OP_RE.match(str(operation))
    return m.group(1).lower() if m else None


@dataclass
class Case:
    path: Path
    root_cause: str      # labelled root-cause service
    fault: str           # cpu, mem, disk, delay, loss, socket
    rep: int
    inject_time: int     # unix seconds

    @classmethod
    def from_path(cls, path: str | Path) -> "Case":
        path = Path(path)
        service, fault = path.parent.name.rsplit("_", 1)
        return cls(
            path=path, root_cause=service, fault=fault, rep=int(path.name),
            inject_time=int((path / "inject_time.txt").read_text().strip()),
        )


def discover_cases(root: str | Path) -> list[Case]:
    """All cases under an extracted RE2-OB directory, in a stable order."""
    return [Case.from_path(p.parent) for p in sorted(Path(root).glob("*_*/*/inject_time.txt"))]


def load_client_calls(traces: pd.DataFrame) -> pd.DataFrame:
    """Reduce raw spans to one row per cross-service call.

    Returns columns: sec (unix seconds), caller, callee, error (bool).

    Calls are taken from *client* spans, whose operation names the callee, not
    by joining server spans to their parents. Under packet loss the server span
    often never exists, so only the client side records the failure; a parent
    join would silently drop exactly the errors R_ij is meant to capture.
    """
    callee = traces["operationName"].map(callee_from_operation)
    caller = traces["serviceName"].replace(_TRACE_NAME_FIX)
    calls = pd.DataFrame({
        "sec": traces["startTime"] // 1_000_000,          # startTime is in µs
        "caller": caller,
        "callee": callee.replace(_TRACE_NAME_FIX),
        "error": traces["statusCode"].fillna(GRPC_OK) != GRPC_OK,
    })
    # Drop non-RPC spans and a service's own server span (caller == callee).
    return calls[calls["callee"].notna() & (calls["caller"] != calls["callee"])]


def window_call_stats(
    calls: pd.DataFrame, start: int, window_s: int
) -> dict[tuple[str, str], tuple[int, int]]:
    """{(caller, callee): (requests, errors)} for calls in [start, start+window_s)."""
    w = calls[(calls["sec"] >= start) & (calls["sec"] < start + window_s)]
    g = w.groupby(["caller", "callee"])["error"].agg(["size", "sum"])
    return {k: (int(n), int(e)) for k, (n, e) in g.iterrows()}


def case_graphs(
    case: Case,
    window_s: int = 60,
    edge_alpha: float = 0.5,
    services: list[str] = OB_SERVICES,
) -> list[tuple[int, CallGraph]]:
    """Build the dynamic call graph for every window of a case.

    Returns (window_start_unix, CallGraph) pairs covering the whole recording.
    """
    traces = pd.read_csv(
        case.path / "traces.csv",
        usecols=["serviceName", "operationName", "startTime", "statusCode"],
    )
    calls = load_client_calls(traces)
    lo, hi = int(calls["sec"].min()), int(calls["sec"].max())
    out = []
    for start in range(lo, hi + 1, window_s):
        stats = window_call_stats(calls, start, window_s)
        out.append((start, build_call_graph(stats, services, edge_alpha=edge_alpha)))
    return out
