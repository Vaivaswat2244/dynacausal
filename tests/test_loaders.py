"""Tests for turning raw RE2 spans into per-window call statistics."""
import pandas as pd

from src.data.loaders import (
    OB_SERVICES, Case, callee_from_operation, load_client_calls, window_call_stats,
)


def spans(rows):
    return pd.DataFrame(rows, columns=["serviceName", "operationName", "startTime", "statusCode"])


def test_callee_parsing():
    assert callee_from_operation("hipstershop.ProductCatalogService/GetProduct") == "productcatalogservice"
    assert callee_from_operation("/hipstershop.ProductCatalogService/ListProducts") == "productcatalogservice"
    assert callee_from_operation("frontend") is None
    assert callee_from_operation("grpc.hipstershop.CurrencyService/Convert") is None  # server span


def test_frontend_name_normalised_to_metric_name():
    c = load_client_calls(spans([
        ["frontendservice", "hipstershop.CartService/GetCart", 1_000_000, 0.0],
    ]))
    assert c.iloc[0]["caller"] == "frontend"
    assert set(c["caller"]) | set(c["callee"]) <= set(OB_SERVICES)


def test_errors_are_non_ok_grpc_codes_and_nan_is_ok():
    c = load_client_calls(spans([
        ["frontendservice", "hipstershop.CartService/GetCart", 1_000_000, 0.0],
        ["frontendservice", "hipstershop.CartService/GetCart", 1_000_000, 14.0],  # UNAVAILABLE
        ["frontendservice", "hipstershop.CartService/GetCart", 1_000_000, 4.0],   # DEADLINE_EXCEEDED
        ["frontendservice", "hipstershop.CartService/GetCart", 1_000_000, None],
    ]))
    assert c["error"].tolist() == [False, True, True, False]


def test_non_rpc_and_self_spans_dropped():
    c = load_client_calls(spans([
        ["frontendservice", "frontend", 1_000_000, None],                          # HTTP entry span
        ["cartservice", "hipstershop.CartService/GetCart", 1_000_000, 0.0],        # self
    ]))
    assert c.empty


def test_window_stats_counts_requests_and_errors_in_range():
    c = load_client_calls(spans([
        ["frontendservice", "hipstershop.CartService/GetCart", 100_000_000, 0.0],
        ["frontendservice", "hipstershop.CartService/GetCart", 159_000_000, 14.0],
        ["frontendservice", "hipstershop.CartService/GetCart", 160_000_000, 14.0],  # next window
    ]))
    assert window_call_stats(c, 100, 60) == {("frontend", "cartservice"): (2, 1)}


def test_case_label_parsed_from_path(tmp_path):
    p = tmp_path / "RE2-OB" / "productcatalogservice_loss" / "1"
    p.mkdir(parents=True)
    (p / "inject_time.txt").write_text("1705342830\n")
    case = Case.from_path(p)
    assert (case.root_cause, case.fault, case.rep, case.inject_time) == (
        "productcatalogservice", "loss", 1, 1705342830)
