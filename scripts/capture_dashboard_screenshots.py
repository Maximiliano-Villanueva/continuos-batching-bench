#!/usr/bin/env python3
"""Capture Streamlit dashboard screenshots for README.

Usage:
  make dashboard   # in another terminal, or let this script use an existing server
  make dashboard-screenshots

Set STREAMLIT_URL if the dashboard is not on http://127.0.0.1:8501
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images" / "dashboard"
URL = os.environ.get("STREAMLIT_URL", "http://127.0.0.1:8501")
OFF_RUN = "740e0566d01c"
ON_RUN = "3f61df7ade09"

TABS = {
    "overview": "Overview",
    "e2-position": "E2 Position",
    "e4-k-sweep": "E4 K sweep",
    "e6-api": "E6 API",
    "e7-speculative": "E7 Speculative",
    "conclusions": "Conclusions",
}


def _wait_for_app(page, timeout_s: float = 90) -> None:
    page.goto(URL, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_selector('[data-testid="stApp"]', timeout=30000)
    page.wait_for_function(
        """() => {
            const app = document.querySelector('[data-testid="stApp"]');
            return app && app.getAttribute('data-test-script-state') === 'running';
        }""",
        timeout=int(timeout_s * 1000),
    )
    # Wait for main title or the empty-state warning (not a traceback)
    page.wait_for_function(
        """() => {
            const text = document.body.innerText;
            if (text.includes('ModuleNotFoundError') || text.includes('Traceback')) return false;
            return text.includes('Continuous Batching Benchmarks')
                || text.includes('No benchmark runs');
        }""",
        timeout=int(timeout_s * 1000),
    )
    page.wait_for_timeout(2000)


def _select_run(page, run_id: str) -> None:
    page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').click()
    page.wait_for_timeout(500)
    page.locator('[role="option"]').filter(has_text=run_id).first.click()
    page.wait_for_timeout(2500)


def _click_tab(page, tab_name: str) -> None:
    page.get_by_role("tab", name=tab_name).click()
    page.wait_for_timeout(2000)


def _screenshot(page, path: Path, *, clip_p99: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if clip_p99:
        # Viewport crop: sidebar + lower main area (p99 chart after scroll)
        page.screenshot(
            path=str(path),
            clip={"x": 0, "y": 0, "width": 1440, "height": 900},
        )
    else:
        page.screenshot(path=str(path), full_page=True)
    print(f"Wrote {path}")


def capture_all() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        _wait_for_app(page)

        _select_run(page, OFF_RUN)
        _click_tab(page, TABS["overview"])
        _screenshot(page, OUT / "01-overview-spec-off.png")

        _click_tab(page, TABS["e4-k-sweep"])
        _screenshot(page, OUT / "02-e4-throughput-spec-off.png")
        page.evaluate(
            """() => {
                const main = document.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTop = main.scrollHeight;
            }"""
        )
        page.wait_for_timeout(1000)
        _screenshot(page, OUT / "02b-e4-p99-spec-off.png", clip_p99=True)

        _select_run(page, ON_RUN)
        _click_tab(page, TABS["e4-k-sweep"])
        _screenshot(page, OUT / "03-e4-throughput-spec-on.png")
        page.evaluate(
            """() => {
                const main = document.querySelector('[data-testid="stMain"]');
                if (main) main.scrollTop = main.scrollHeight;
            }"""
        )
        page.wait_for_timeout(1000)
        _screenshot(page, OUT / "03b-e4-p99-spec-on.png", clip_p99=True)

        _select_run(page, OFF_RUN)
        _click_tab(page, TABS["e2-position"])
        _screenshot(page, OUT / "04-e2-position.png")

        _click_tab(page, TABS["e6-api"])
        _screenshot(page, OUT / "05-e6-api.png")

        _click_tab(page, TABS["e7-speculative"])
        _screenshot(page, OUT / "06-e7-spec-off.png")

        _select_run(page, ON_RUN)
        _click_tab(page, TABS["e7-speculative"])
        _screenshot(page, OUT / "07-e7-spec-on.png")

        _select_run(page, OFF_RUN)
        _click_tab(page, TABS["conclusions"])
        _screenshot(page, OUT / "08-conclusions.png")

        browser.close()


def main() -> None:
    capture_all()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "Ensure the dashboard is running: make dashboard "
            "(or STREAMLIT_URL=http://127.0.0.1:8502 make dashboard-screenshots)",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
