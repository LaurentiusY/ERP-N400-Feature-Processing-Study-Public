"""Preprocessing summary across subjects (stages 1 and 3).

One row per subject from ``preproc.json`` (trials, ICA, interpolation, behaviour), and the
channel sets of D4 resolved on the recorded channel names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .features import channel_matrix
from .paths import deriv_dir
from .study import CHANNEL_SETS

BEHAVIOUR_KEYS = (
    "n_no_response",
    "accuracy_related",
    "accuracy_unrelated",
    "median_rt_related",
    "median_rt_unrelated",
)


def preproc_row(meta: dict) -> dict:
    """The fields of one ``preproc.json`` that the stage reports need."""
    behaviour = meta.get("behaviour", {})
    return {
        "subject": meta["subject"],
        "n_targets": meta["n_targets"],
        "n_kept": meta["n_kept"],
        "n_related": meta["n_related"],
        "n_unrelated": meta["n_unrelated"],
        "n_dropped_epoching": meta["n_dropped_epoching"],
        "n_dropped_autoreject": meta["n_dropped_autoreject"],
        "n_dropped_unrepairable": meta.get("n_dropped_unrepairable", 0),
        "n_dropped_threshold": meta.get("n_dropped_threshold", 0),
        "n_dropped_flat": meta.get("n_dropped_flat", 0),
        "ica_eog_n": meta["ica_eog"]["n_components"],
        "ica_eog_var": meta["ica_eog"]["explained_variance"],
        "ica_muscle_n": meta["ica_muscle"]["n_components"],
        "ica_muscle_var": meta["ica_muscle"]["explained_variance"],
        "interpolated_fraction": meta["interpolated_fraction"],
        "n_channels": len(meta["channels"]),
        **{key: behaviour.get(key) for key in BEHAVIOUR_KEYS},
    }


def read_preproc(subjects: list[str], root: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Rows of the subjects that have a ``preproc.json``, and each subject's channel list."""
    root = root or deriv_dir()
    rows, channels = [], {}
    for subject in subjects:
        path = root / subject / "preproc.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        rows.append(preproc_row(meta))
        channels[subject] = list(meta["channels"])
    if not rows:
        raise FileNotFoundError(f"no preproc.json for {subjects} under {root}")
    return pd.DataFrame(rows), channels


def channel_sets(ch_names: list[str], allow_missing: bool = False) -> dict[str, list[str]]:
    """Channels (virtual channels for C5) of every D4 level on these channel names.

    A level that needs a channel the recording lacks raises ``KeyError``, unless
    ``allow_missing`` (PURSUE has no FCz, so C5 is unavailable there): then it maps to ``[]``.
    """
    out = {}
    for code in CHANNEL_SETS:
        try:
            out[code] = channel_matrix(code, ch_names)[1]
        except KeyError:
            if not allow_missing:
                raise
            out[code] = []
    return out
