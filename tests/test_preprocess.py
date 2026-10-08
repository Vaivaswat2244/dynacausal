"""Tests for feature extraction, using small synthetic case directories."""
import numpy as np
import pandas as pd
import pytest

from src.data.loaders import Case
from src.data.preprocess import (
    D_IN, FEATURE_NAMES, LOG_LEVELS, case_features, log_features,
    metric_features, min_max_per_series, trace_features,
)

T0 = 15 * 113_333_334  # unix time divisible by 15, so it aligns with the bucket grid


@pytest.fixture
def case_dir(tmp_path):
    """Two services, 60 s of data, fault at +30 s."""
    d = tmp_path / "svc-a_cpu" / "1"
    d.mkdir(parents=True)
    (d / "inject_time.txt").write_text(str(T0 + 30))

    secs = list(range(T0, T0 + 60))
    pd.DataFrame({
        "time": secs,
        "svc-a_cpu": [1.0] * 30 + [9.0] * 30,   # jumps at the fault
        "svc-b_cpu": [2.0] * 60,                 # constant
        "svc-a_mem": range(60),                  # ramp
    }).to_csv(d / "simple_metrics.csv", index=False)

    pd.DataFrame({
        "timestamp": [(T0 + s) * 1_000_000_000 for s in (1, 2, 16, 40)],
        "container_name": ["svc-a", "svc-a", "svc-b", "svc-a"],
        "level": ["INFO", "error", None, "info"],
    }).to_csv(d / "logs.csv", index=False)

    pd.DataFrame({
        "serviceName": ["svc-a", "svc-a", "frontendservice"],
        "operationName": ["op", "op", "frontend"],
        "startTime": [(T0 + 1) * 1_000_000, (T0 + 2) * 1_000_000, (T0 + 1) * 1_000_000],
        "duration": [100, 300, 50],
        "statusCode": [0.0, 14.0, None],
        "traceID": ["t1", "t2", "t3"], "spanID": ["s1", "s2", "s3"],
        "parentSpanID": ["", "", ""], "methodName": ["m", "m", "m"],
        "time": ["x", "x", "x"], "startTimeMillis": [0, 0, 0],
    }).to_csv(d / "traces.csv", index=False)
    return d


SERVICES = ["svc-a", "svc-b", "frontend"]


def grid():
    return np.arange(T0, T0 + 60, 15)


class TestMetricFeatures:
    def test_bucket_means(self, case_dir):
        out = metric_features(case_dir, SERVICES, grid(), 15)
        cpu = FEATURE_NAMES.index("metric_cpu")  # same order within families
        # svc-a cpu: buckets of 15 x 1.0, then 15 x 1.0, then 9.0s
        assert out[0, 0, 0] == 1.0 and out[0, 3, 0] == 9.0

    def test_missing_family_is_zero(self, case_dir):
        out = metric_features(case_dir, SERVICES, grid(), 15)
        # svc-b has no mem column -> zeros
        assert np.all(out[1, :, 1] == 0)


class TestLogFeatures:
    def test_levels_folded_and_counted(self, case_dir):
        out = log_features(case_dir, SERVICES, grid(), 15)
        info, err, other = (LOG_LEVELS.index(l) for l in ("info", "error", "other"))
        assert out[0, 0, info] == 1      # "INFO" case-folded
        assert out[0, 0, err] == 1
        assert out[1, 1, other] == 1     # NaN level -> other
        assert out[0, 2, info] == 1      # the +40 s line lands in bucket 2


class TestTraceFeatures:
    def test_count_duration_error(self, case_dir):
        out = trace_features(case_dir, SERVICES, grid(), 15)
        n, dur, err = range(3)
        assert out[0, 0, n] == 2
        assert out[0, 0, dur] == 200     # mean(100, 300)
        assert out[0, 0, err] == 0.5     # one of two spans failed

    def test_frontendservice_mapped_to_frontend(self, case_dir):
        out = trace_features(case_dir, SERVICES, grid(), 15)
        assert out[2, 0, 0] == 1         # under "frontend", not dropped

    def test_nan_status_is_not_error(self, case_dir):
        out = trace_features(case_dir, SERVICES, grid(), 15)
        assert out[2, 0, 2] == 0.0


class TestNormalisation:
    def test_unit_range_and_constant_to_zero(self):
        x = np.stack([np.stack([np.arange(4.0), np.full(4, 7.0)], axis=1)])  # (1,4,2)
        out = min_max_per_series(x)
        assert np.allclose(out[0, :, 0], [0, 1/3, 2/3, 1])
        assert np.all(out[0, :, 1] == 0)


class TestCaseFeatures:
    def test_shapes_and_bounds(self, case_dir):
        case = Case.from_path(case_dir)
        g, feats = case_features(case, grid_s=15, services=SERVICES)
        assert feats.shape == (3, len(g), D_IN)
        assert feats.min() >= 0.0 and feats.max() <= 1.0

    def test_fault_jump_survives_normalisation(self, case_dir):
        case = Case.from_path(case_dir)
        g, feats = case_features(case, grid_s=15, services=SERVICES)
        cpu = 0
        assert feats[0, :, cpu].max() - feats[0, :, cpu].min() > 0.9
