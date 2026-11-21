"""Fixed preprocessing (docs/engineering_guide.md section 4).

Kessler et al.'s Luck path for N400, ``ica_ica_None_0.1_P9P10_None_200ms_int``, in the order
of their ``src/1-pre-multiverse.py`` and ``src/multiverse.py`` (github.com/kesslerr/m4d,
CC BY 4.0; the ICA and autoreject helpers follow their ``src/utils.py``).

Outputs in ``<deriv>/sub-XXX/``: ``clean-raw.fif`` (continuous, for the envelopes),
``waveform-epo.fif``, ``interp_labels.npy`` (kept trials x EEG channels of ``clean-raw.fif``,
True = interpolated by autoreject), ``trials.tsv`` (kept trials, epoch order) and
``preproc.json``.

Autoreject only interpolates, as in Kessler's "int" version, with one exception decided at
stage 1: a trial whose every channel is flagged has nothing to interpolate from and would be
all zero, so it is dropped (``n_dropped_unrepairable`` in ``preproc.json``).

The Draschkow et al. (2018) tasks (guide section 13.3) differ in reading the data and events
(:mod:`n400fp.draschkow`), in the EOG channels (SO1/SO2 as recorded) and in the reference: the
data are already referenced to the linked mastoids, so the preliminary Cz reference and the
final re-reference are skipped (``preproc_config``). After gate 1 (guide section 13.8) both
tasks keep only the 28 recorded ERP CORE electrodes in the saved epochs (ICA and interpolation
still use every channel), and the sentence task skips the muscle ICA and drops trials above a
global peak-to-peak threshold instead of interpolating channels. The path specified before
gate 1 (``N400_PREPROC=prespecified``, guide section 13.11: all channels, muscle ICA and
autoreject interpolation in both tasks) is kept for the sensitivity analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ERPCORE_ELECTRODES, LCD_DELAY, RELATED, UNRELATED, PreprocConfig
from .events import _norm, behaviour_summary, target_trials
from .paths import dataset_name, deriv_dir, preproc_variant, subject_deriv
from .pursue import DISPLAY_DELAY as PURSUE_DISPLAY_DELAY
from .pursue import ELECTRODES as PURSUE_ELECTRODES
from .pursue import REFERENCE as PURSUE_REFERENCE

log = logging.getLogger(__name__)

EVENT_ID = {"related": 1, "unrelated": 2}
SKIP_BY_ANNOTATION = ("edge", "bad_acq_skip", "boundary")
OUTPUT_FILES = (
    "clean-raw.fif",
    "waveform-epo.fif",
    "interp_labels.npy",
    "trials.tsv",
    "preproc.json",
)


DATASET_SETTINGS = {
    "erpcore": {},
    "scene": {"initial_reference": (), "reference": (), "keep_channels": ERPCORE_ELECTRODES},
    "sentence": {
        "initial_reference": (),
        "reference": (),
        "keep_channels": ERPCORE_ELECTRODES,
        "muscle_ica": False,
        "bad_trials": "reject",
    },
    # ERP CORE path; TP9/TP10 final reference, the 27 ERP CORE electrodes present (14.3)
    "pursue": {"reference": PURSUE_REFERENCE, "keep_channels": PURSUE_ELECTRODES},
}
# Draschkow tasks as specified before gate 1 (guide 13.3): only the re-referencing is skipped
PRESPECIFIED_SETTINGS = {
    "scene": {"initial_reference": (), "reference": ()},
    "sentence": {"initial_reference": (), "reference": ()},
}


def preproc_config(
    dataset: str = "erpcore", variant: str | None = None, **overrides
) -> PreprocConfig:
    """The fixed preprocessing settings of a dataset (guide sections 4, 13.3 and 13.8), or with
    ``variant="prespecified"`` the Draschkow path before gate 1 (13.11; default: ``N400_PREPROC``).
    """
    variant = variant or preproc_variant()
    if variant == "main":
        settings = DATASET_SETTINGS[dataset]
    elif variant == "prespecified" and dataset in PRESPECIFIED_SETTINGS:
        settings = PRESPECIFIED_SETTINGS[dataset]
    else:
        raise ValueError(f"no preprocessing variant {variant!r} for {dataset!r}")
    return PreprocConfig(**{**settings, **overrides})


def kept_channels(ch_names: list[str], cfg: PreprocConfig) -> list[str]:
    """``cfg.keep_channels`` as spelled in ``ch_names``, in recording order (all if empty)."""
    if not cfg.keep_channels:
        return list(ch_names)
    wanted = {c.upper() for c in cfg.keep_channels}
    found = [c for c in ch_names if c.upper() in wanted]
    if len(found) != len(wanted):
        missing = wanted - {c.upper() for c in found}
        raise ValueError(f"channels not in the recording: {sorted(missing)}")
    return found


def is_done(subject: str, root: Path | None = None) -> bool:
    folder = (root or deriv_dir()) / subject
    return all((folder / f).exists() for f in OUTPUT_FILES)


def read_raw(path: Path):
    import mne

    return mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec="utf-8", verbose="error")


def keep_targets(raw, delay: float = LCD_DELAY) -> pd.DataFrame:
    """Behaviour table from all events, then keep only target (and boundary) annotations.

    Targets are renamed ``related``/``unrelated`` and shifted by the display delay (ERP CORE:
    the 26 ms LCD delay). Returns the table of all targets (``target_trials``), in the order of
    the remaining annotations.
    """
    import mne

    ann = raw.annotations
    trials = target_trials(ann.onset, ann.description, delay)
    onsets, durations, descriptions = [], [], []
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description, strict=True):
        code = _norm(desc)
        if code in RELATED or code in UNRELATED:
            onsets.append(onset + delay)
            durations.append(duration)
            descriptions.append("related" if code in RELATED else "unrelated")
        elif str(desc).lower().startswith("boundary"):
            onsets.append(onset)
            durations.append(duration)
            descriptions.append("boundary")
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions, orig_time=ann.orig_time))
    return trials


def recalculate_eog(raw) -> None:
    """Bipolar HEOG (left - right) and VEOG (lower - FP2), as Kessler's recalculate_eog_signal."""
    import mne

    names = {c.lower(): c for c in raw.ch_names}
    get = lambda name: raw.get_data(picks=[names[name.lower()]])  # noqa: E731
    heog = get("HEOG_left") - get("HEOG_right")
    veog = get("VEOG_lower") - get("FP2")
    info = mne.create_info(["HEOG", "VEOG"], raw.info["sfreq"], "eog")
    raw.add_channels([mne.io.RawArray(np.vstack([heog, veog]), info, verbose="error")], True)
    raw.drop_channels([names[n.lower()] for n in ("HEOG_left", "HEOG_right", "VEOG_lower")])


def set_montage(raw) -> None:
    import mne

    rename = {c: c.capitalize() for c in raw.ch_names if c.upper() in ("FP1", "FP2")}
    raw.rename_channels(rename)
    # MNE 1.13 renamed "standard_1005" to "colin27_1005" (same positions); older MNE has only
    # the former, which is removed in MNE 1.14
    name = "colin27_1005"
    if name not in mne.channels.get_builtin_montages():
        name = "standard_1005"
    raw.set_montage(mne.channels.make_standard_montage(name), on_missing="ignore")


def ica_remove(raw, kind: str, cfg: PreprocConfig) -> tuple[int, float]:
    """Fit ICA on a 1 Hz high-passed copy, remove EOG or muscle components from ``raw``."""
    from mne.preprocessing import ICA

    filt = raw.copy().filter(
        l_freq=cfg.ica_highpass, h_freq=None, n_jobs=cfg.n_jobs, verbose="error"
    )
    ica = ICA(
        n_components=cfg.ica_n_components,
        max_iter="auto",
        method=cfg.ica_method,
        random_state=cfg.ica_random_state,
    )
    ica.fit(filt, verbose="error")
    if kind == "eog":
        indices, _ = ica.find_bads_eog(raw, verbose="error")
    elif kind == "muscle":
        indices, _ = ica.find_bads_muscle(raw, verbose="error")
    else:
        raise ValueError(kind)
    indices = [int(i) for i in indices]
    var = 0.0
    if indices:
        var = float(
            ica.get_explained_variance_ratio(filt, components=indices, ch_type="eeg")["eeg"]
        )
    ica.exclude = indices
    ica.apply(raw, verbose="error")
    return len(indices), var


def autoreject_interpolate(epochs, cfg: PreprocConfig):
    """Kessler's ``autorej(version="int")``: interpolate bad channels per trial.

    Returns the repaired epochs, the boolean matrix of interpolated channels for the kept
    epochs, and the boolean mask of epochs that autoreject dropped.
    """
    from autoreject import AutoReject

    n_ch = len(epochs.ch_names)
    rng = np.random.RandomState(cfg.autoreject_subsample_seed)
    n = len(epochs)
    sample = rng.choice(n, int(n * cfg.autoreject_subsample), replace=False)
    seed = cfg.autoreject_seed
    while True:
        ar = AutoReject(
            n_interpolate=[n_ch],
            consensus=[n_ch],
            random_state=seed,
            n_jobs=cfg.n_jobs,
            cv=cfg.autoreject_cv,
            verbose=False,
        )
        ar.fit(epochs[sample].copy())
        repaired, reject_log = ar.transform(epochs, return_log=True)
        if len(repaired) / n >= cfg.autoreject_min_kept:
            break
        seed += 1
    labels = np.nan_to_num(np.asarray(reject_log.labels, dtype=float))
    dropped = np.asarray(reject_log.bad_epochs, dtype=bool)
    if list(reject_log.ch_names) != list(repaired.ch_names):
        raise RuntimeError("autoreject channel order differs from the epochs")
    interp = labels[~dropped] == 2
    return repaired, interp, dropped


def reject_trials(epochs, cfg: PreprocConfig):
    """Drop flat trials, then the trials whose peak-to-peak amplitude on a kept channel exceeds
    a global threshold estimated by autoreject (``get_rejection_threshold``, cross-validated)
    on the remaining trials.

    A trial is flat when the median over the kept channels of its per-channel SD is below
    ``cfg.flat_sd``: a partly failed recording (sentence sub-017, guide section 13.8) otherwise
    pulls the threshold down to a few microvolts, so that only the flat trials are kept.

    Returns the kept epochs, the (all False) interpolation matrix, the boolean mask of dropped
    epochs, the threshold in volts and the mask of the flat epochs.
    """
    from autoreject import get_rejection_threshold

    picked = epochs.copy().pick(kept_channels(epochs.ch_names, cfg))
    data = picked.get_data()
    flat = np.median(data.std(axis=2), axis=1) < cfg.flat_sd
    threshold = get_rejection_threshold(
        picked[np.flatnonzero(~flat)], ch_types="eeg", random_state=cfg.autoreject_seed,
        cv=cfg.autoreject_cv, verbose=False,
    )["eeg"]  # fmt: skip
    dropped = flat | (np.ptp(data, axis=2).max(axis=1) > threshold)
    repaired = epochs.copy().drop(np.flatnonzero(dropped), reason="BAD_TRIAL", verbose="error")
    interp = np.zeros((len(repaired), len(epochs.ch_names)), dtype=bool)
    return repaired, interp, dropped, float(threshold), flat


def unrepairable(interp: np.ndarray) -> np.ndarray:
    """Trials whose every channel is flagged: with no good channel to interpolate from,
    autoreject sets them to zero, so they are dropped (the only trials that are)."""
    return interp.all(axis=1)


def prepare_events(raw, dataset: str) -> pd.DataFrame:
    """Keep only the target annotations (``related``/``unrelated``); returns the trial table."""
    if dataset == "erpcore":
        return keep_targets(raw)
    if dataset == "pursue":
        return keep_targets(raw, PURSUE_DISPLAY_DELAY)
    from . import draschkow

    return draschkow.keep_targets(raw, dataset)


def prepare_channels(raw, dataset: str) -> dict:
    """EOG channels, reference conversion and montage after resampling."""
    if dataset == "erpcore":
        recalculate_eog(raw)
        set_montage(raw)
        return {}
    if dataset == "pursue":
        from . import pursue

        pursue.recalculate_eog(raw)
        set_montage(raw)
        return {}
    from . import draschkow

    converted = draschkow.to_linked_mastoids(raw)
    set_montage(raw)
    return {"converted_to_linked_mastoids": converted}


def preprocess_raw(
    raw,
    subject: str,
    out_dir: Path,
    cfg: PreprocConfig = PreprocConfig(),
    dataset: str = "erpcore",
    extra: dict | None = None,
) -> dict:
    """Run section 4 on a loaded raw recording.

    ERP CORE: all channels typed EEG, as read from EEGLAB. Draschkow tasks: BrainVision with
    SO1/SO2 typed EOG (:func:`n400fp.draschkow.read_raw`); use ``preproc_config(dataset)``.
    ``extra`` is added to ``preproc.json``.
    """
    import mne

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = prepare_events(raw, dataset)
    raw.resample(cfg.sfreq, n_jobs=cfg.n_jobs, verbose="error")
    channel_info = prepare_channels(raw, dataset)
    if cfg.initial_reference:
        raw.set_eeg_reference(list(cfg.initial_reference), projection=False, verbose="error")

    n_eog, var_eog = ica_remove(raw, "eog", cfg)
    raw.pick("eeg")
    n_emg, var_emg = ica_remove(raw, "muscle", cfg) if cfg.muscle_ica else (0, 0.0)
    raw.filter(
        l_freq=cfg.highpass,
        h_freq=None,
        method="fir",
        fir_design="firwin",
        skip_by_annotation=SKIP_BY_ANNOTATION,
        n_jobs=cfg.n_jobs,
        verbose="error",
    )
    if cfg.reference:
        raw.set_eeg_reference(list(cfg.reference), projection=False, verbose="error")
    raw.save(out_dir / "clean-raw.fif", overwrite=True, verbose="error")

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="error")
    if len(events) != len(trials) or not np.array_equal(events[:, 2] - 1, trials.label.values):
        raise RuntimeError(f"{subject}: events and behaviour table do not match")
    # a target whose envelope window (env_tmin, earlier than epoch_tmin) starts before the
    # recording has a waveform epoch but no envelope epoch: dropped from both (guide 14.9)
    onset_s = (events[:, 0] - raw.first_samp) / raw.info["sfreq"]
    too_early = np.flatnonzero(onset_s + cfg.env_tmin < 0)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=cfg.epoch_tmin,
        tmax=cfg.epoch_tmax,
        baseline=cfg.baseline,
        detrend=None,
        proj=False,
        reject_by_annotation=False,
        preload=True,
        verbose="error",
    )
    edge = np.flatnonzero(np.isin(epochs.selection, too_early))
    if edge.size:
        epochs.drop(edge, reason="ENVELOPE_WINDOW", verbose="error")
    selection = np.asarray(epochs.selection)
    threshold, flat = None, np.zeros(len(epochs), dtype=bool)
    if cfg.bad_trials == "interpolate":
        repaired, interp, dropped = autoreject_interpolate(epochs, cfg)
    elif cfg.bad_trials == "reject":
        repaired, interp, dropped, threshold, flat = reject_trials(epochs, cfg)
    else:
        raise ValueError(f"bad_trials must be 'interpolate' or 'reject', not {cfg.bad_trials!r}")
    kept = selection[~dropped]
    empty = unrepairable(interp)
    unrepairable_targets = trials.event_index.to_numpy()[kept[empty]].tolist()
    if empty.any():
        repaired.drop(np.flatnonzero(empty), reason="NO_GOOD_CHANNEL", verbose="error")
        interp, kept = interp[~empty], kept[~empty]
    all_channels = list(repaired.ch_names)
    saved = kept_channels(all_channels, cfg)
    repaired.pick(saved)
    repaired.save(out_dir / "waveform-epo.fif", overwrite=True, verbose="error")
    np.save(out_dir / "interp_labels.npy", interp)

    kept_trials = trials.iloc[kept].reset_index(drop=True)
    kept_trials.insert(0, "epoch_index", np.arange(len(kept_trials)))
    kept_trials.to_csv(out_dir / "trials.tsv", sep="\t", index=False)

    meta = {
        "subject": subject,
        "dataset": dataset,
        "n_targets": int(len(trials)),
        "n_kept": int(len(kept_trials)),
        "n_related": int((kept_trials.label == 0).sum()),
        "n_unrelated": int((kept_trials.label == 1).sum()),
        "n_dropped_epoching": int(len(trials) - len(selection)),  # includes the next one
        "n_dropped_envelope_window": int(edge.size),
        "n_dropped_autoreject": int(dropped.sum()) if threshold is None else 0,
        "n_dropped_threshold": int((dropped & ~flat).sum()) if threshold is not None else 0,
        "n_dropped_flat": int(flat.sum()),
        "rejection_threshold_uv": None if threshold is None else threshold * 1e6,
        "n_dropped_unrepairable": len(unrepairable_targets),
        "unrepairable_targets": unrepairable_targets,  # event_index of the dropped targets
        "ica_eog": {"n_components": n_eog, "explained_variance": var_eog},
        "ica_muscle": {"n_components": n_emg, "explained_variance": var_emg},
        "interpolated_fraction": (
            float(interp[:, [all_channels.index(c) for c in saved]].mean()) if interp.size else 0.0
        ),  # over the saved channels
        "interpolated_fraction_by_channel": dict(
            zip(all_channels, map(float, interp.mean(axis=0)), strict=True)
        ),
        "channels": list(repaired.ch_names),
        "behaviour": behaviour_summary(trials) if dataset in ("erpcore", "pursue") else {},
        **channel_info,
        **(extra or {}),
        "config": asdict(cfg),
        "versions": _versions(),
    }
    (out_dir / "preproc.json").write_text(json.dumps(meta, indent=2))
    if meta["n_related"] != meta["n_unrelated"]:
        log.warning(
            "%s: unequal class sizes %s/%s", subject, meta["n_related"], meta["n_unrelated"]
        )
    return meta


def preprocess_subject(
    subject: str,
    cfg: PreprocConfig | None = None,
    data_root: Path | None = None,
    deriv_root: Path | None = None,
    overwrite: bool = False,
) -> dict | None:
    """Preprocess one subject of the current dataset (``N400_DATASET``)."""
    dataset = dataset_name()
    cfg = cfg or preproc_config(dataset)
    if not overwrite and is_done(subject, deriv_root):
        log.info("%s: preprocessing exists, skipped", subject)
        return None
    extra = {}
    if dataset == "erpcore":
        from .download import find_raw_file

        raw = read_raw(find_raw_file(subject, data_root))
    elif dataset == "pursue":  # data_root: the BIDS folder (default paths.pursue_dir())
        from . import pursue

        raw = pursue.read_raw(pursue.raw_path(subject, data_root))
        extra["subject_code"] = pursue.subject_code(subject, data_root)
    else:  # data_root: the extracted Draschkow folder (default paths.draschkow_dir())
        from . import draschkow

        raw = draschkow.read_raw(draschkow.raw_path(subject, dataset, data_root))
        extra["subject_code"] = draschkow.subject_code(subject, data_root)
    out_dir = subject_deriv(subject, deriv_root)
    return preprocess_raw(raw, subject, out_dir, cfg, dataset, extra)


def _versions() -> dict:
    import importlib

    out = {}
    for name in ("mne", "autoreject", "picard", "sklearn", "numpy", "scipy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out
