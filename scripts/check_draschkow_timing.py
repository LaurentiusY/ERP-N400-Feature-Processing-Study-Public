"""Quality check of the Draschkow target times against the authors' epochs (guide 13.6, G-2).

For every participant and task, the targets computed by :mod:`n400fp.draschkow` (scene: object
onset; sentence: critical-word onset) are matched to the authors' epochs (``epo/*-epo.fif``,
their kept trials at 200 Hz). Pass: every author epoch has a target within 10 ms with the same
condition. Reads markers and epoch events only; no EEG signal is analysed.

    python scripts/check_draschkow_timing.py [--root DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from n400fp import draschkow
from n400fp.paths import draschkow_dir

AUTHOR_LABEL = {111: 0, 112: 1, 200: 0, 201: 0, 210: 1, 211: 1}
TOLERANCE = 0.010  # s


def check(code: str, task: str, root: Path) -> dict:
    import mne

    raw = mne.io.read_raw_brainvision(
        root / "unprocessed" / f"{code}_{draschkow.TASK_FILES[task]}.vhdr",
        preload=False,
        verbose="error",
    )
    lookup = draschkow.lookup_onsets(root) if task == "sentence" else None
    ours = draschkow.target_table(raw.annotations.onset, raw.annotations.description, task, lookup)
    ep = mne.read_epochs(
        root / "epo" / f"{code}-{draschkow.TASK_FILES[task]}-epo.fif",
        preload=False,
        verbose="error",
    )
    times = ep.events[:, 0] / ep.info["sfreq"]
    nearest = np.abs(ours.onset.to_numpy()[None, :] - times[:, None]).argmin(axis=1)
    diff = np.abs(ours.onset.to_numpy()[nearest] - times)
    same = ours.label.to_numpy()[nearest] == np.array([AUTHOR_LABEL[c] for c in ep.events[:, 2]])
    return {
        "code": code,
        "task": task,
        "n_ours": len(ours),
        "n_ours_0": int((ours.label == 0).sum()),
        "n_ours_1": int((ours.label == 1).sum()),
        "n_author": len(times),
        "median_ms": 1000 * float(np.median(diff)),
        "max_ms": 1000 * float(diff.max()),
        "labels_agree": float(same.mean()),
        "passed": bool((diff <= TOLERANCE).all() and same.all()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args()
    root = args.root or draschkow_dir()
    rows = [check(c, t, root) for c in draschkow.subject_codes(root) for t in draschkow.TASK_FILES]
    df = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df.round(1).to_string(index=False))
    for task, g in df.groupby("task"):
        print(
            f"\n{task}: {g.passed.sum()}/{len(g)} passed; trials per class "
            f"{g.n_ours_0.min()}-{g.n_ours_0.max()} / {g.n_ours_1.min()}-{g.n_ours_1.max()}; "
            f"max |difference| {g.max_ms.max():.1f} ms; labels agree {g.labels_agree.min():.0%}"
        )
    if not df.passed.all():
        raise SystemExit("timing check failed")


if __name__ == "__main__":
    main()
