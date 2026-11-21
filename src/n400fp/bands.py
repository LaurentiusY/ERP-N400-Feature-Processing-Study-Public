"""Band envelopes (docs/engineering_guide.md section 5).

Per band: FIR band-pass on the continuous clean data, Hilbert analytic signal, epochs from
-0.5 s to the waveform epochs' last sample (no baseline) around the kept targets, the same
per-trial channel interpolation as the waveform epochs applied to the analytic signal, then
its modulus (the amplitude envelope), restricted to the channels of the waveform epochs
(``PreprocConfig.keep_channels``). Saved as linear amplitude in ``env-<band>-epo.fif``;
the log transform and baseline correction happen per trial in :mod:`n400fp.features`.

Filtering and the Hilbert transform run on the continuous data and the interpolation is
spatial (sample by sample), so the epochs need no margin: longer epochs cropped afterwards
give identical values, but can run past the end of the recording.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BAND_ORDER, BANDS, PreprocConfig
from .paths import deriv_dir
from .preprocess import EVENT_ID, SKIP_BY_ANNOTATION, kept_channels

log = logging.getLogger(__name__)


def envelope_file(subject: str, band: str, root: Path | None = None) -> Path:
    return (root or deriv_dir()) / subject / f"env-{band}-epo.fif"


def is_done(subject: str, root: Path | None = None) -> bool:
    return all(envelope_file(subject, b, root).exists() for b in BAND_ORDER)


def interpolate_per_trial(epochs, interp: np.ndarray):
    """Interpolate, trial by trial, the channels flagged in ``interp`` (trials x channels).

    Trials with the same set of flagged channels are interpolated together.
    """
    import mne

    data = epochs.get_data(copy=True)
    names = np.asarray(epochs.ch_names)
    patterns: dict[tuple[str, ...], list[int]] = {}
    for i, row in enumerate(interp):
        if row.any():
            patterns.setdefault(tuple(names[row]), []).append(i)
    for bads, idx in patterns.items():
        sub = epochs[idx].copy()
        sub.info["bads"] = list(bads)
        sub.interpolate_bads(reset_bads=True, verbose="error")
        data[idx] = sub.get_data(copy=True)
    return mne.EpochsArray(
        data,
        epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        baseline=None,
        verbose="error",
    )


def interpolated_envelope(epochs, interp: np.ndarray):
    """Amplitude envelope of analytic-signal epochs after the per-trial interpolation.

    The real and imaginary parts are interpolated separately and the modulus is taken
    afterwards. Interpolation (across channels) and the Hilbert transform (along time) are
    both linear, so this equals the envelope of the interpolated band-passed voltage and is
    never negative. Interpolating the envelope itself goes below zero (the spline weights
    have both signs), which the log transform turns into extreme outliers.
    """
    import mne

    data = epochs.get_data(copy=True)
    parts = []
    for part in (data.real, data.imag):
        real = mne.EpochsArray(
            part,
            epochs.info,
            events=epochs.events,
            tmin=epochs.tmin,
            event_id=epochs.event_id,
            baseline=None,
            verbose="error",
        )
        parts.append(interpolate_per_trial(real, interp).get_data(copy=True))
    return mne.EpochsArray(
        np.hypot(*parts),
        epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        baseline=None,
        verbose="error",
    )


def compute_envelopes(
    subject: str,
    cfg: PreprocConfig = PreprocConfig(),
    deriv_root: Path | None = None,
    overwrite: bool = False,
) -> None:
    import mne

    root = deriv_root or deriv_dir()
    folder = root / subject
    if not overwrite and is_done(subject, root):
        log.info("%s: envelopes exist, skipped", subject)
        return
    raw = mne.io.read_raw_fif(folder / "clean-raw.fif", preload=True, verbose="error")
    trials = pd.read_csv(folder / "trials.tsv", sep="\t")
    interp = np.load(folder / "interp_labels.npy")
    wave = mne.read_epochs(folder / "waveform-epo.fif", preload=False, verbose="error")

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="error")
    events = events[trials.event_index.to_numpy()]
    if not np.array_equal(events[:, 2] - 1, trials.label.to_numpy()):
        raise RuntimeError(f"{subject}: envelope events do not match trials.tsv")

    for band in BAND_ORDER:
        lo, hi = BANDS[band]
        filt = raw.copy().filter(
            l_freq=lo,
            h_freq=hi,
            method="fir",
            fir_design="firwin",
            skip_by_annotation=SKIP_BY_ANNOTATION,
            n_jobs=cfg.n_jobs,
            verbose="error",
        )
        filt.apply_hilbert(envelope=False, n_jobs=cfg.n_jobs, verbose="error")
        epochs = mne.Epochs(
            filt,
            events,
            event_id=EVENT_ID,
            tmin=cfg.env_tmin,
            tmax=wave.tmax,  # same last sample as the waveform epochs (0.8 s is off the grid)
            baseline=None,
            proj=False,
            reject_by_annotation=False,
            preload=True,
            verbose="error",
        )
        if len(epochs) != len(trials):
            raise RuntimeError(f"{subject}/{band}: {len(epochs)} epochs for {len(trials)} trials")
        epochs = interpolated_envelope(epochs, interp)
        epochs.pick(kept_channels(epochs.ch_names, cfg))
        if epochs.ch_names != wave.ch_names:
            raise RuntimeError(f"{subject}/{band}: channel order differs from the waveform")
        epochs.save(envelope_file(subject, band, root), overwrite=True, verbose="error")
        log.info("%s: %s envelope saved", subject, band)
