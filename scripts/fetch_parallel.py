"""Download a large file with parallel HTTP range requests (resumable), then check its MD5.

Zenodo limits the speed of each connection, so a 27 GB file over one connection can take days.
The file is split into parts that are fetched concurrently; every part resumes from what is
already on disk, so the command can be rerun after an interruption. Standard library only.

Zenodo drops long connections and rate-limits bursts, so every failure is retried: dropped or
truncated responses resume where they stopped, HTTP 429/503 wait for ``Retry-After``, other
errors back off exponentially with jitter, and connections are opened a few seconds apart.

    python scripts/fetch_parallel.py URL OUT --size BYTES --md5 HEX [--parts 64] [--workers 12]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 1 << 20
REPORT_EVERY = 60  # seconds
BACKOFF_START, BACKOFF_MAX = 5, 600  # seconds
STAGGER = 3  # seconds between the first requests of the workers
USER_AGENT = "n400fp-fetch/1.0 (research data download; python-urllib)"
RETRYABLE = (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError)


class Progress:
    def __init__(self, total: int, done: int):
        self.total, self.done, self.lock = total, done, threading.Lock()
        self.t0, self.done0 = time.time(), done

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def line(self) -> str:
        rate = (self.done - self.done0) / max(time.time() - self.t0, 1e-9)
        left = (self.total - self.done) / rate if rate > 0 else float("inf")
        return (
            f"{self.done / 1e9:.2f}/{self.total / 1e9:.2f} GB "
            f"({100 * self.done / self.total:.1f}%), {rate / 1e6:.2f} MB/s, "
            f"about {left / 3600:.1f} h left"
        )


def retry_after(err: BaseException) -> float | None:
    """Seconds asked for by an HTTP 429/503 ``Retry-After`` header, if any."""
    if isinstance(err, urllib.error.HTTPError) and err.code in (429, 503):
        try:
            return float(err.headers.get("Retry-After", 60))
        except ValueError:
            return 60.0
    return None


def fetch_part(
    url: str, path: Path, start: int, end: int, progress: Progress, first_wait: float = 0
) -> None:
    """Bytes ``start..end`` (inclusive) into ``path``, resuming and retrying until complete."""
    want = end - start + 1
    delay = BACKOFF_START
    time.sleep(first_wait)
    while True:
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return
        headers = {"Range": f"bytes={start + have}-{end}", "User-Agent": USER_AGENT}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status != 206:
                    raise http.client.HTTPException(f"range request ignored (HTTP {resp.status})")
                with open(path, "ab") as f:  # bytes read before a dropped connection are kept
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        progress.add(len(chunk))
            delay = BACKOFF_START
        except RETRYABLE as err:
            wait = retry_after(err)
            if wait is None:
                wait = delay * random.uniform(0.8, 1.2)
                delay = min(delay * 2, BACKOFF_MAX)
            print(f"   {path.name}: {err!r}; retry in {wait:.0f} s", flush=True)
            time.sleep(wait)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(64 * CHUNK):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("url")
    p.add_argument("out", type=Path)
    p.add_argument("--size", type=int, required=True, help="file size in bytes")
    p.add_argument("--md5", help="expected MD5 (hex); checked after assembly")
    p.add_argument("--parts", type=int, default=64)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    out = args.out
    if out.exists() and out.stat().st_size == args.size:
        print(f"{out} already complete")
    else:
        step = -(-args.size // args.parts)
        ranges = [(k * step, min((k + 1) * step, args.size) - 1) for k in range(args.parts)]
        paths = [out.with_name(f"{out.name}.part{k:03d}") for k in range(args.parts)]
        progress = Progress(args.size, sum(q.stat().st_size for q in paths if q.exists()))
        stop = threading.Event()

        def report() -> None:
            while not stop.wait(REPORT_EVERY):
                print("  ", progress.line(), flush=True)

        threading.Thread(target=report, daemon=True).start()
        print(f"downloading {args.size / 1e9:.2f} GB in {args.parts} parts, "
              f"{args.workers} connections", flush=True)  # fmt: skip
        with ThreadPoolExecutor(args.workers) as pool:
            waits = [STAGGER * k if k < args.workers else 0 for k in range(args.parts)]
            jobs = [
                pool.submit(fetch_part, args.url, q, s, e, progress, w)
                for q, (s, e), w in zip(paths, ranges, waits, strict=True)
            ]
            for job in jobs:
                job.result()
        stop.set()
        print("  ", progress.line(), flush=True)
        with open(out, "wb") as f:
            for q in paths:
                with open(q, "rb") as part:
                    while block := part.read(64 * CHUNK):
                        f.write(block)
        if out.stat().st_size != args.size:
            raise SystemExit(f"assembled size {out.stat().st_size} != {args.size}")
        for q in paths:
            q.unlink()
    if args.md5:
        got = md5_of(out)
        if got != args.md5.lower():
            raise SystemExit(f"MD5 mismatch: {got} != {args.md5}")
        print(f"MD5 OK ({got})")


if __name__ == "__main__":
    main()
