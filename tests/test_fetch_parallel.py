"""scripts/fetch_parallel.py against a local server that rate-limits and drops connections."""

import hashlib
import importlib.util
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_parallel.py"
PAYLOAD = os.urandom(300_000)


def load_script():
    spec = importlib.util.spec_from_file_location("fetch_parallel", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlakyRangeHandler(BaseHTTPRequestHandler):
    """Answers the first request with 429, then drops every other range response halfway."""

    lock = threading.Lock()
    count = 0
    user_agents: list[str] = []

    def do_GET(self) -> None:
        with self.lock:
            FlakyRangeHandler.count += 1
            n = FlakyRangeHandler.count
            self.user_agents.append(self.headers.get("User-Agent", ""))
        if n == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        lo, hi = (int(x) for x in self.headers["Range"].removeprefix("bytes=").split("-"))
        body = PAYLOAD[lo : hi + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{len(PAYLOAD)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if n % 2 == 0 and len(body) > 1:
            self.wfile.write(body[: len(body) // 2])  # truncated: client sees IncompleteRead
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FlakyRangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/file"
    httpd.shutdown()


def test_download_survives_rate_limit_and_dropped_connections(server, tmp_path, monkeypatch):
    fp = load_script()
    monkeypatch.setattr(fp, "BACKOFF_START", 0)
    monkeypatch.setattr(fp, "STAGGER", 0)
    out = tmp_path / "data.bin"
    md5 = hashlib.md5(PAYLOAD).hexdigest()
    argv = [str(SCRIPT), server, str(out), "--size", str(len(PAYLOAD)), "--md5", md5]
    monkeypatch.setattr(sys, "argv", [*argv, "--parts", "4", "--workers", "2"])

    fp.main()

    assert out.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("*.part*"))
    assert FlakyRangeHandler.count > 5  # the 429 and the dropped responses were retried
    assert all(ua.startswith("n400fp-fetch/") for ua in FlakyRangeHandler.user_agents)


def test_retry_after_only_for_rate_limits():
    fp = load_script()
    headers = {"Retry-After": "7"}
    limited = fp.urllib.error.HTTPError("u", 429, "Too Many Requests", headers, None)
    missing = fp.urllib.error.HTTPError("u", 404, "Not Found", headers, None)
    assert fp.retry_after(limited) == 7
    assert fp.retry_after(missing) is None
    assert fp.retry_after(TimeoutError()) is None
