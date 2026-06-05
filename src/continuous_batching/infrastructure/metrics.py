from __future__ import annotations

import httpx


def metrics_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/metrics"


def fetch_prometheus_metrics(base_url: str, timeout_s: float = 5.0) -> str | None:
    """Fetch vLLM Prometheus metrics if the endpoint is enabled."""
    url = metrics_url(base_url)
    try:
        resp = httpx.get(url, timeout=timeout_s)
        if resp.status_code == 200:
            return resp.text
    except Exception:  # noqa: BLE001
        return None
    return None
