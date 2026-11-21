"""The PURSUE N400 word-pair dataset: a preregistered replication (guide section 14).

NEMAR ``on007052`` v1.0.0 (OpenNeuro ``ds007052``, CC0; Couperus, Bukach & Reed): 288
participants from three undergraduate institutions (``sub-1xxx``, ``sub-2xxx``, ``sub-3xxx``),
the ERP CORE N400 paradigm with the same event codes (targets 211/212 and 221/222, responses
201/202; practice trials 110/120/210/220 are not targets). BIDS, one EEGLAB ``.set`` per
participant in ``sub-XXXX/eeg/sub-XXXX_task-WordPR_eeg.set`` below :func:`n400fp.paths.pursue_dir`.

Recording: BrainVision actiCHamp at 500 Hz, 29 EEG electrodes (TP9/TP10 instead of P9/P10, no
FCz) and three EOG electrodes (HEOG R, HEOG L, and the lower VEOG electrode, labelled
"VEOG L" in the data and "VEOG Lower" in the authors' channel operations, channel 32, which they
pair with FP2). Preregistered adaptations of the ERP CORE path (14.3): no display delay; HEOG =
HEOG L - HEOG R and VEOG = VEOG L(ower) - FP2; Cz as the initial and TP9/TP10 as the final
reference; the 27 ERP CORE electrodes present are kept.

Participants are numbered ``sub-001`` ... ``sub-288`` in the sorted order of the labels in the
dataset's ``participants.tsv`` (so a partial download cannot shift the numbering); three
recordings contain no EEG data and are excluded a priori (14.6).
"""

from __future__ import annotations

from pathlib import Path

from .config import ERPCORE_ELECTRODES
from .paths import pursue_dir, subject_number

TASK = "WordPR"
N_PARTICIPANTS = 288
EMPTY_RECORDINGS = ("2002", "2025", "3042")  # .set files without EEG data (19 kB)
DISPLAY_DELAY = 0.0  # s; not reported for PURSUE (ERP CORE's 26 ms is specific to its monitor)
EOG_PAIRS = {"HEOG": ("HEOG L", "HEOG R"), "VEOG": ("VEOG L", "FP2")}  # VEOG L = lower VEOG
REFERENCE = ("TP9", "TP10")
ELECTRODES = tuple(c for c in ERPCORE_ELECTRODES if c != "FCz")  # FCz was not recorded
MIN_TRIALS_PER_CLASS = 30  # post-processing exclusion (14.6)
SITES = {"1": "site 1 (sub-1xxx)", "2": "site 2 (sub-2xxx)", "3": "site 3 (sub-3xxx)"}


def subject_codes(root: Path | None = None) -> list[str]:
    """Participant labels (without ``sub-``) of ``participants.tsv`` in sorted order;
    ``sub-001`` is the first."""
    path = (root or pursue_dir()) / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"no PURSUE participants.tsv at {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    codes = sorted(line.split()[0].removeprefix("sub-") for line in lines[1:] if line.strip())
    if len(codes) != N_PARTICIPANTS or len(set(codes)) != N_PARTICIPANTS:
        raise ValueError(f"{path}: expected {N_PARTICIPANTS} distinct participants")
    return codes


def subject_code(subject: str, root: Path | None = None) -> str:
    return subject_codes(root)[subject_number(subject) - 1]


def excluded_a_priori(subject: str, root: Path | None = None) -> bool:
    return subject_code(subject, root) in EMPTY_RECORDINGS


def site(code: str) -> str:
    return SITES[code[0]]


def raw_path(subject: str, root: Path | None = None) -> Path:
    root = root or pursue_dir()
    code = subject_code(subject, root)
    return root / f"sub-{code}" / "eeg" / f"sub-{code}_task-{TASK}_eeg.set"


def read_raw(path: Path):
    """All channels typed EEG, as for ERP CORE; the EOG pairs are rebuilt by
    :func:`recalculate_eog`."""
    import mne

    return mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec="utf-8", verbose="error")


def recalculate_eog(raw) -> None:
    """Bipolar HEOG (HEOG L - HEOG R) and VEOG (VEOG L, the lower electrode, - FP2); the
    monopolar EOG channels are dropped (FP2 is kept)."""
    import mne
    import numpy as np

    names = {c.lower(): c for c in raw.ch_names}
    get = lambda name: raw.get_data(picks=[names[name.lower()]])  # noqa: E731
    data = [get(a) - get(b) for a, b in EOG_PAIRS.values()]
    info = mne.create_info(list(EOG_PAIRS), raw.info["sfreq"], "eog")
    raw.add_channels([mne.io.RawArray(np.vstack(data), info, verbose="error")], True)
    raw.drop_channels([names[n.lower()] for n in ("HEOG L", "HEOG R", "VEOG L")])
