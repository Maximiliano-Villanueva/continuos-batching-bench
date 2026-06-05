from continuous_batching.infrastructure.metrics import metrics_url


def test_metrics_url_strips_v1_suffix():
    assert metrics_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"


def test_metrics_url_without_v1():
    assert metrics_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/metrics"
