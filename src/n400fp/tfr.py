"""Complex Morlet coefficients for the time-frequency sensitivity analysis (guide 13.10).

On the continuous clean data, one annotation-free segment at a time (``edge``, ``boundary``
and ``bad_acq_skip`` split the recording, as in the band filtering of section 5): complex
Morlet wavelets at the 12 frequencies of :data:`n400fp.config.TFR_FREQS` (3-10 cycles,
zero-mean). Epochs from -0.5 s to the waveform epochs' last sample around the kept targets,
the per-trial channel interpolation applied to the real and imaginary parts separately, the
channels of the waveform epochs kept (a window that crosses a boundary takes each sample from
its own segment). Saved as ``tfr-epo.npz``: ``coef`` (complex64, trials x
frequencies x channels x times), ``freqs``, ``times`` and ``ch_names``. The log amplitude,
its baseline correction and the phase features happen per trial in :mod:`n400fp.features`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .bands import interpolate_per_trial
from .config import TFR_CYCLES, TFR_FREQS, PreprocConfig
from .paths import deriv_dir
from .preprocess import EVENT_ID, SKIP_BY_ANNOTATION, kept_channels

log = logging.getLogger(__name__)


def tfr_file(subject: str, root: Path | None = None) -> Path:
    return (root or deriv_dir()) / subject / "tfr-epo.npz"


def segments(raw) -> list[tuple[int, int]]:
    """``(start, stop)`` sample indices of the parts of ``raw`` between skip annotations."""
    sfreq, n = raw.info["sfreq"], raw.n_times
    cuts = []
    for onset, duration, desc in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description, strict=True
    ):
        if desc.lower().startswith(SKIP_BY_ANNOTATION):
            start = int(round((onset - raw.first_time) * sfreq))
            cuts.append((start, start + int(round(duration * sfreq))))
    out, pos = [], 0
    for start, stop in sorted(cuts):
        if start > pos:
            out.append((pos, min(start, n)))
        pos = max(pos, stop)
    if pos < n:
        out.append((pos, n))
    return out


def morlet_segment(data: np.ndarray, sfreq: float, freq: float, n_cycles: float) -> np.ndarray:
    """Complex Morlet coefficients ``(channels, times)`` of one continuous segment."""
    from mne.time_frequency import tfr_array_morlet

    coef = tfr_array_morlet(
        data[None],
        sfreq,
        freqs=[freq],
        n_cycles=[n_cycles],
        zero_mean=True,
        use_fft=True,
        output="complex",
        verbose="error",
    )
    return coef[0, :, 0]


def fill_epochs(z: np.ndarray, seg: np.ndarray, lo: int, first: np.ndarray) -> int:
    """Copy the part of each epoch window that lies in segment ``seg`` (starting at sample
    ``lo``) into ``z`` (trials x channels x times); ``first`` = first sample of each window.

    Returns how many windows start before the segment (they cross its first boundary)."""
    n_times, hi = z.shape[-1], lo + seg.shape[-1]
    partial = 0
    for i, f in enumerate(first):
        a, b = max(f, lo), min(f + n_times, hi)
        if a >= b:
            continue
        z[i, :, a - f : b - f] = seg[:, a - lo : b - lo]
        partial += a > f
    return partial


def max_sigma_t() -> float:
    """Temporal SD (s) of the widest wavelet (the lowest frequency)."""
    return max(n / (2 * np.pi * f) for f, n in zip(TFR_FREQS, TFR_CYCLES, strict=True))


def edge_windows(first: np.ndarray, n_times: int, segs, margin: int) -> int:
    """Windows that start or end within ``margin`` samples of the edge of their segment."""
    count = 0
    for f in first:
        for lo, hi in segs:
            if lo <= f < hi or lo < f + n_times <= hi:
                count += (f - lo < margin) or (hi - (f + n_times) < margin)
                break
    return count


def _one_frequency(subject, data, info, segs, first, n_times, interp, picks, freq, n_cycles):
    """``(trials, kept channels, times)`` complex64 coefficients at one frequency."""
    import mne

    z = np.full((len(first), data.shape[0], n_times), np.nan, np.complex128)
    crossing = 0
    for lo, hi in segs:
        if not ((first < hi) & (first + n_times > lo)).any():
            continue
        try:
            seg = morlet_segment(data[:, lo:hi], info["sfreq"], freq, n_cycles)
        except ValueError as err:  # MNE: wavelet longer than the signal
            raise RuntimeError(
                f"{subject}: segment of {(hi - lo) / info['sfreq']:.1f} s holding epoch windows "
                f"is too short for the {freq:.2f} Hz wavelet"
            ) from err
        crossing += fill_epochs(z, seg, lo, first)
    if np.isnan(z).any():  # samples inside a skipped stretch (bad_acq_skip)
        raise RuntimeError(f"{subject}: epoch windows overlap a skipped part of the data")
    if crossing and freq == TFR_FREQS[0]:
        log.warning("%s: %d epoch windows cross a segment boundary", subject, crossing)
    parts = []
    for part in (z.real, z.imag):
        ep = mne.EpochsArray(part, info, tmin=0.0, baseline=None, verbose="error")
        parts.append(interpolate_per_trial(ep, interp).get_data(copy=True)[:, picks])
    log.info("%s: %.2f Hz done", subject, freq)
    return (parts[0] + 1j * parts[1]).astype(np.complex64)


def compute_tfr(
    subject: str,
    cfg: PreprocConfig = PreprocConfig(),
    deriv_root: Path | None = None,
    overwrite: bool = False,
) -> None:
    import mne

    root = deriv_root or deriv_dir()
    folder = root / subject
    if not overwrite and tfr_file(subject, root).exists():
        log.info("%s: time-frequency coefficients exist, skipped", subject)
        return
    raw = mne.io.read_raw_fif(folder / "clean-raw.fif", preload=True, verbose="error")
    trials = pd.read_csv(folder / "trials.tsv", sep="\t")
    interp = np.load(folder / "interp_labels.npy")
    wave = mne.read_epochs(folder / "waveform-epo.fif", preload=False, verbose="error")

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="error")
    events = events[trials.event_index.to_numpy()]
    if not np.array_equal(events[:, 2] - 1, trials.label.to_numpy()):
        raise RuntimeError(f"{subject}: time-frequency events do not match trials.tsv")

    sfreq = raw.info["sfreq"]
    start = int(round(cfg.env_tmin * sfreq))
    stop = int(round(wave.tmax * sfreq))  # same last sample as the waveform epochs
    times = np.arange(start, stop + 1) / sfreq
    first = events[:, 0] - raw.first_samp + start
    if first.min() < 0 or first.max() + len(times) > raw.n_times:
        raise RuntimeError(f"{subject}: an epoch window leaves the recording")
    segs = segments(raw)
    n_edge = edge_windows(first, len(times), segs, int(5 * max_sigma_t() * sfreq))
    if n_edge:
        msg = "%s: %d epoch windows near a segment edge (2 Hz wavelet partly on zero padding)"
        log.warning(msg, subject, n_edge)
    data = raw.get_data()
    kept = kept_channels(raw.ch_names, cfg)
    if kept != wave.ch_names:
        raise RuntimeError(f"{subject}: channel order differs from the waveform")
    picks = [raw.ch_names.index(c) for c in kept]
    args = (subject, data, raw.info, segs, first, len(times), interp, picks)
    per_freq = Parallel(n_jobs=cfg.n_jobs, backend="threading")(
        delayed(_one_frequency)(*args, freq, n_cycles)
        for freq, n_cycles in zip(TFR_FREQS, TFR_CYCLES, strict=True)
    )
    coef = np.stack(per_freq, axis=1)

    path = tfr_file(subject, root)
    tmp = path.with_name("tfr-epo.tmp.npz")  # written fully, then renamed
    np.savez(tmp, coef=coef, freqs=np.asarray(TFR_FREQS), times=times, ch_names=np.asarray(kept))
    tmp.replace(path)
    log.info("%s: time-frequency coefficients saved %s", subject, coef.shape)
