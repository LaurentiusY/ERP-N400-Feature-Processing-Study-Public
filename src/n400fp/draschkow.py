"""The Draschkow et al. (2018) dataset: visual-scene and auditory-sentence tasks (guide section 13).

Zenodo 10.5281/zenodo.1421347 (CC BY 4.0); code github.com/DejanDraschkow/n3n4. 40 participants
did both tasks in one session; one BrainVision recording per participant and task in
``unprocessed/<code>_<vis_scene|sentence>.vhdr`` below :func:`n400fp.paths.draschkow_dir`.

* ``scene``: object-scene consistency; markers 111 (consistent) and 112 (inconsistent) at the
  object onset; no display delay is applied (none is reported).
* ``sentence``: spoken German sentences; markers 200/201 (congruent, low/high cloze) and 210/211
  (incongruent) at the sentence onset. The critical word starts at marker + 255 ms (audio
  delay) + the word's ``onset`` in ``csv/lookUp320_2.csv``; the sentence number is the code of
  the preceding stimulus marker minus 1. This is the rule of the authors' preprocessing
  notebook; checked against their epochs (median 1 ms, max 6 ms at their 200 Hz).

Class labels follow the rest of the package: 0 = expected (consistent/congruent), 1 =
unexpected (inconsistent/incongruent); the annotations use the package's class names
``related``/``unrelated`` for 0/1.

Channels: 62-channel recordings are referenced online to the linked mastoids; 64-channel
recordings also contain M1 and M2, which are converted to the same linked-mastoid reference and
dropped. SO1 and SO2 are the EOG channels.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .events import TRIAL_COLUMNS
from .paths import draschkow_dir, subject_number

TASK_FILES = {"scene": "vis_scene", "sentence": "sentence"}
SCENE_CODES = {111: 0, 112: 1}
SENTENCE_CODES = {200: 0, 201: 0, 210: 1, 211: 1}
CONDITIONS = {
    "scene": ("consistent", "inconsistent"),
    "sentence": ("congruent", "incongruent"),
}
SENTENCE_DELAY = 0.255  # s, audio delay in the authors' code
EOG_CHANNELS = ("SO1", "SO2")
MASTOIDS = ("M1", "M2")
LOOKUP = Path("csv") / "lookUp320_2.csv"
CLASS_NAMES = ("related", "unrelated")  # annotation names of labels 0 and 1

_STIMULUS = re.compile(r"^Stimulus/\D*(\d+)\s*$")


def subject_codes(root: Path | None = None) -> list[str]:
    """Participant codes in alphabetical order; ``sub-001`` is the first."""
    folder = (root or draschkow_dir()) / "unprocessed"
    codes = sorted({p.stem.rsplit("_", 2)[0] for p in folder.glob("*_sentence.vhdr")})
    if not codes:
        raise FileNotFoundError(f"no Draschkow recordings in {folder}")
    return codes


def subject_code(subject: str, root: Path | None = None) -> str:
    return subject_codes(root)[subject_number(subject) - 1]


def raw_path(subject: str, task: str, root: Path | None = None) -> Path:
    root = root or draschkow_dir()
    return root / "unprocessed" / f"{subject_code(subject, root)}_{TASK_FILES[task]}.vhdr"


def read_raw(path: Path):
    import mne

    return mne.io.read_raw_brainvision(path, eog=EOG_CHANNELS, preload=True, verbose="error")


def marker_codes(descriptions) -> np.ndarray:
    """Numeric code of each ``Stimulus/...`` annotation; -1 for anything else."""
    out = []
    for desc in descriptions:
        m = _STIMULUS.match(str(desc))
        out.append(int(m.group(1)) if m else -1)
    return np.array(out, dtype=int)


@lru_cache(maxsize=4)
def _lookup(path: Path) -> pd.Series:
    table = pd.read_csv(path, index_col=0)
    table = table[table.label.str.startswith(("con_", "inc_"))].copy()
    table["item"] = table.label.str.rsplit("_", n=1).str[-1].astype(int)
    return table.set_index(["trig", "item"]).onset


def lookup_onsets(root: Path | None = None) -> pd.Series:
    """Critical-word onset (s, from the audio start) by (condition code, sentence number)."""
    return _lookup((root or draschkow_dir()) / LOOKUP)


def target_table(onsets, descriptions, task: str, lookup: pd.Series | None = None) -> pd.DataFrame:
    """One row per target in temporal order, with the package's trial columns plus ``item``.

    ``onset`` is the target time (scene: object onset; sentence: critical-word onset);
    reaction times and correctness do not exist in these tasks and are left empty.
    """
    onsets = np.asarray(onsets, dtype=float)
    codes = marker_codes(descriptions)
    table = SCENE_CODES if task == "scene" else SENTENCE_CODES
    stimulus = np.flatnonzero(codes >= 0)
    rows = []
    for i in np.flatnonzero(np.isin(codes, list(table))):
        code, item, t = int(codes[i]), -1, onsets[i]
        if task == "sentence":
            before = stimulus[stimulus < i]
            if not len(before):
                raise ValueError(f"sentence marker {code} at {t:.3f} s has no preceding item code")
            item = int(codes[before[-1]]) - 1
            if lookup is None or (code, item) not in lookup.index:
                raise KeyError(f"no lookup onset for condition {code}, sentence {item}")
            t = t + SENTENCE_DELAY + float(lookup[(code, item)])
        label = table[code]
        rows.append((t, code, CONDITIONS[task][label], label, item))
    rows.sort()
    out = pd.DataFrame(
        {
            "event_index": np.arange(len(rows)),
            "onset": [r[0] for r in rows],
            "trigger": [str(r[1]) for r in rows],
            "condition": [r[2] for r in rows],
            "label": [r[3] for r in rows],
            "rt_ms": np.nan,
            "correct": np.nan,
            "item": [r[4] for r in rows],
        }
    )
    return out[[*TRIAL_COLUMNS, "item"]]


def keep_targets(raw, task: str, root: Path | None = None) -> pd.DataFrame:
    """Replace the annotations by the targets (``related``/``unrelated`` = label 0/1)."""
    import mne

    ann = raw.annotations
    lookup = lookup_onsets(root) if task == "sentence" else None
    trials = target_table(ann.onset, ann.description, task, lookup)
    raw.set_annotations(
        mne.Annotations(
            trials.onset.to_numpy(),
            np.zeros(len(trials)),
            [CLASS_NAMES[lab] for lab in trials.label],
            orig_time=ann.orig_time,
        )
    )
    return trials


def to_linked_mastoids(raw) -> bool:
    """64-channel recordings: re-reference to (M1 + M2) / 2 and drop M1, M2.

    62-channel recordings are already referenced online to the linked mastoids. Returns whether
    the data were re-referenced.
    """
    names = {c.upper(): c for c in raw.ch_names}
    if not all(m in names for m in MASTOIDS):
        return False
    mastoids = [names[m] for m in MASTOIDS]
    raw.set_eeg_reference(mastoids, projection=False, verbose="error")
    raw.drop_channels(mastoids)
    return True
