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


def test_missing_latency_families_zero(live_run, tmp_path):
    """Families with no live source must come out as all-zero features, not NaN."""
    case_dir = prepare_live_case(live_run, tmp_path / "prep")
    sample = build_sample(Case.from_path(case_dir), grid_s=15)
    for fam in ("metric_latency-50", "metric_latency-90"):
        col = sample["feats"][:, :, FEATURE_NAMES.index(fam)]
        assert float(col.abs().sum()) == 0.0
