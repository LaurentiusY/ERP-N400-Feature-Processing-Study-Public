"""Download every file of a NEMAR dataset manifest, verifying each against the manifest.

The manifest (``https://data.nemar.org/<id>/<version>/manifest.json``) lists every file with
its size and checksum: sha256 for annexed files (the EEG data), the git blob hash for the
others. A file is downloaded to ``<name>.part``, checked, and only then renamed; files already
present and correct are skipped, so the command can be rerun after an interruption. Transient
errors (dropped connections, HTTP 429/5xx) are retried with exponential backoff. Exits with
status 1 if any file is still missing or wrong at the end. Standard library only.

    python scripts/fetch_manifest.py MANIFEST_URL OUT_DIR [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHUNK = 1 << 20
ATTEMPTS = 8
BACKOFF_START, BACKOFF_MAX = 5, 300  # seconds
USER_AGENT = "n400fp-fetch/1.0 (research data download; python-urllib)"
RETRYABLE = (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError)


def checksum(path: Path, algorithm: str) -> str:
    """sha256 of the file, or its git blob hash (sha1 of ``blob <size>\\0`` + content)."""
    if algorithm == "sha256":
        h = hashlib.sha256()
    elif algorithm == "git":
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"unknown checksum algorithm {algorithm!r}")
    with open(path, "rb") as f:
        while block := f.read(64 * CHUNK):
            h.update(block)
    return h.hexdigest()


def is_valid(path: Path, entry: dict) -> bool:
    return (
        path.exists()
        and path.stat().st_size == entry["size"]
        and checksum(path, entry["checksum_algorithm"]) == entry["checksum"]
    )


def fetch(entry: dict, out: Path) -> str | None:
    """Download one manifest entry; returns an error message, or None when the file is valid."""
    target = out / entry["path"]
    if is_valid(target, entry):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    delay = BACKOFF_START
    for attempt in range(1, ATTEMPTS + 1):
        part.unlink(missing_ok=True)
        req = urllib.request.Request(entry["url"], headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as f:
                while chunk := resp.read(CHUNK):
                    f.write(chunk)
            if is_valid(part, entry):
                part.replace(target)
                return None
            problem = f"size {part.stat().st_size} / checksum differ from the manifest"
        except RETRYABLE as err:
            problem = repr(err)
        if attempt < ATTEMPTS:
            wait = delay * random.uniform(0.8, 1.2)
            print(f"   {entry['path']}: {problem}; retry {attempt} in {wait:.0f} s", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, BACKOFF_MAX)
    part.unlink(missing_ok=True)
    return f"{entry['path']}: {problem}"


def load_manifest(source: str) -> list[dict]:
    if Path(source).exists():
        return json.loads(Path(source).read_text())
    req = urllib.request.Request(source, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("manifest", help="manifest URL or local file")
    p.add_argument("out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    entries = load_manifest(args.manifest)
    total = sum(e["size"] for e in entries)
    print(f"{len(entries)} files, {total / 1e9:.2f} GB -> {args.out}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    errors, done = [], 0
    with ThreadPoolExecutor(args.workers) as pool:
        for err in pool.map(lambda e: fetch(e, args.out), entries):
            done += 1
            if err:
                errors.append(err)
            if done % 50 == 0 or done == len(entries):
                print(f"   {done}/{len(entries)} files checked, {len(errors)} failed", flush=True)
    if errors:
        print("FAILED:", *errors, sep="\n  ")
        return 1
    print(f"all {len(entries)} files present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
