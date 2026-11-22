"""Live-cluster runs (phase 2) loading through the D1 pipeline."""
import json

import pandas as pd
import pytest

from src.data.dataset import build_sample
from src.data.live import prepare_live_case
from src.data.loaders import OB_SERVICES, Case
from src.data.preprocess import FEATURE_NAMES


@pytest.fixture
def live_run(tmp_path):
    """Tiny live run: 60 s at 1 s, fault at t=30, redis named redis-cart."""
    run = tmp_path / "runs" / "productcatalogservice_loss" / "1"
    run.mkdir(parents=True)
    t = list(range(1000, 1060))
    pd.DataFrame({
        "time": t,
        "frontend_container-cpu-usage-seconds-total": [0.5] * 30 + [0.9] * 30,
        "redis-cart_container-memory-working-set-bytes": [5e6] * 60,
        "frontend_container-network-receive-bytes-total": range(60),
    }).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame({
        "time": [1001], "timestamp": [1001 * 10**9],
        "container_name": ["redis-cart"], "message": ["m"], "level": ["info"],
    }).to_csv(run / "logs.csv", index=False)
    pd.DataFrame({
        "time": [1001, 1002],
        "traceID": ["a", "b"], "spanID": ["s1", "s2"],
        "serviceName": ["frontend", "redis-cart"],
        "methodName": ["Convert", "x"],
        "operationName": ["hipstershop.CurrencyService/Convert", "GET"],
        "startTimeMillis": [1001000, 1002000],
        "startTime": [1001 * 10**6, 1002 * 10**6],
        "duration": [100, 100], "statusCode": [14, None], "parentSpanID": ["", ""],
    }).to_csv(run / "traces.csv", index=False)
    (run / "inject_time.txt").write_text("1030")
    (run / "run.json").write_text(json.dumps({"service": "productcatalogservice"}))
    return run


def test_prepared_case_loads_as_d1(live_run, tmp_path):
    """A prepared live run must go through build_sample with D1 shapes and names."""
    case_dir = prepare_live_case(live_run, tmp_path / "prep")
    sample = build_sample(Case.from_path(case_dir), grid_s=15)

    n, _, d = sample["feats"].shape
    assert (n, d) == (len(OB_SERVICES), len(FEATURE_NAMES))
    assert sample["label"] == OB_SERVICES.index("productcatalogservice")
    assert sample["inject_idx"] == 3  # grid starts at 990; 1030 -> bucket 1035

    # redis-cart renamed: its logs land on the redis row
    redis = OB_SERVICES.index("redis")
    assert sample["feats"][redis, :, FEATURE_NAMES.index("log_info")].sum() > 0
    # client span becomes the frontend -> currencyservice edge
    f, c = OB_SERVICES.index("frontend"), OB_SERVICES.index("currencyservice")
    pairs = set(zip(sample["call_src"].tolist(), sample["call_dst"].tolist()))
    assert (f, c) in pairs
    assert int(sample["call_err"].sum()) == 1  # statusCode 14 counted as error


def test_service_nobody_calls_has_zero_latency(live_run, tmp_path):
    """A service with no spans calling it gets all-zero latency features, not NaN."""
    case_dir = prepare_live_case(live_run, tmp_path / "prep")
    sample = build_sample(Case.from_path(case_dir), grid_s=15)
    ad = OB_SERVICES.index("adservice")
    for fam in ("metric_latency-50", "metric_latency-90"):
        col = sample["feats"][ad, :, FEATURE_NAMES.index(fam)]
        assert float(col.abs().sum()) == 0.0


def test_scale_with_stats_uses_given_range_not_the_data():
    """Fixed lo/hi: values past the fitted range leave [0,1]; zero range -> 0."""
    import numpy as np
    from src.data.preprocess import scale_with_stats

    x = np.array([[[5.0, 7.0], [20.0, 7.0]]])        # 1 service, 2 buckets, 2 feats
    lo, hi = np.array([[0.0, 3.0]]), np.array([[10.0, 3.0]])
    out = scale_with_stats(x, lo, hi)
    assert out[0, :, 0].tolist() == [0.5, 2.0]       # not clipped, not refitted
    assert out[0, :, 1].tolist() == [0.0, 0.0]       # zero-range feature


def test_causal_scale_ignores_the_window_and_the_future():
    """Only the history sets the range; clip keeps the model's 0..1 input range."""
    import numpy as np
    from src.data.preprocess import causal_scale

    hist = np.array([[[0.0, 5.0], [10.0, 5.0]]])      # feat 0 ranges 0..10, feat 1 constant 5
    win = np.array([[[5.0, 5.0], [30.0, 9.0]]])
    out = causal_scale(win, hist, clip=True)
    assert out[0, :, 0].tolist() == [0.5, 1.0]        # 30 is past the history max -> 1
    assert out[0, :, 1].tolist() == [0.0, 1.0]        # constant history: equal -> 0, above -> 1
    raw = causal_scale(win, hist, clip=False)
    assert raw[0, :, 0].tolist() == [0.5, 3.0]        # unclipped keeps the overshoot
    assert raw[0, :, 1].tolist() == [0.0, 0.0]        # zero range -> 0, as in Phase 1


def test_latency_columns_from_calls_to_the_service():
    """Latency of S comes from spans that call S, in seconds, NaN -> ffill -> 0."""
    from src.data.live import latency_columns

    times = pd.Series(range(990, 1035))               # buckets 990, 1005, 1020
    tr = pd.DataFrame({
        "serviceName": ["frontend", "frontend", "currencyservice", "frontend"],
        "operationName": ["hipstershop.CurrencyService/Convert"] * 2
                         + ["grpc.hipstershop.CurrencyService/Convert", "GET"],
        "startTime": [1006 * 10**6, 1007 * 10**6, 1006 * 10**6, 1021 * 10**6],
        "duration": [2_000_000, 4_000_000, 9_000_000, 500_000],
    })
    val, present = latency_columns(tr, times)
    cur = val["currencyservice_latency-50"]
    assert cur[times < 1005].eq(0).all()              # nothing before the first span
    assert cur[(times >= 1005) & (times < 1020)].eq(3.0).all()   # median of 2 s and 4 s; server span (9 s) not used
    assert cur[times >= 1020].eq(3.0).all()           # forward filled ...
    assert not present["currencyservice_latency-50"][times >= 1020].any()   # ... and marked absent
    assert val["frontend_latency-50"][times >= 1020].eq(0.5).all()         # frontend entry span


def test_impact_flag_needs_a_real_rise(tmp_path):
    """p95 up 50% or error share up 0.05 flags the run; a flat run is not flagged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("impact_check", "scripts/impact_check.py")
    ic = importlib.util.module_from_spec(spec); spec.loader.exec_module(ic)

    def run(name, fault_dur, fault_codes):
        d = tmp_path / name / "1"; d.mkdir(parents=True)
        secs = list(range(850, 1000)) + list(range(1000, 1100))
        pd.DataFrame({
            "serviceName": "frontend", "operationName": "GET",
            "startTime": [s * 10**6 for s in secs],
            "duration": [1000] * 150 + [fault_dur] * 100,
            "statusCode": [None] * 150 + fault_codes,
        }).to_csv(d / "traces.csv", index=False)
        (d / "run.json").write_text(json.dumps({"inject_time": 1000, "fault_end": 1100}))
        return ic.impact_of(d)

    assert run("a_cpu", 1000, [None] * 100)["impactful"] is False
    assert run("b_cpu", 1500, [None] * 100)["reason"] == ["p95_rose_50pct"]
    assert run("c_cpu", 1000, [14] * 5 + [None] * 95)["reason"] == ["err_share_rose_0.05"]
    assert run("d_cpu", 1400, [14] * 4 + [None] * 96)["impactful"] is False
