"""Feature extraction before normalization (section 6.1, steps 1-4).

Order: component selection (D1) -> per-trial log transform and baseline correction of the
envelopes -> window and 50 ms bin averages (D2) -> channel selection or left-right
symmetrization (D4). The result is a 4-D array ``(trials, components, channels, bins)``;
normalization (D3) and classification happen in :mod:`n400fp.decode` inside the folds.
Nothing here uses statistics across trials, so it can run before cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BAND_ORDER, FeatureConfig
from .paths import deriv_dir
from .study import (
    CHANNEL_SETS,
    MIDLINE,
    REFERENCE_CHANNELS,
    SYMMETRIC_PAIRS,
    WINDOWS,
    Config,
    components,
)

TIME_TOL = 1e-6


@dataclass
class SubjectData:
    """Waveform (uV) and linear band envelopes of one subject's kept trials."""

    subject: str
    ch_names: list[str]
    y: np.ndarray  # 0 related, 1 unrelated
    wave: np.ndarray  # (trials, channels, times), microvolts
    wave_times: np.ndarray
    env: dict[str, np.ndarray] = field(default_factory=dict)  # band -> (trials, channels, times)
    env_times: np.ndarray | None = None
    trials: pd.DataFrame | None = None
    tfr: np.ndarray | None = None  # complex Morlet, (trials, frequencies, channels, times)
    tfr_times: np.ndarray | None = None


def load_subject(
    subject: str, root: Path | None = None, envelopes: bool = True, tfr: bool = False
) -> SubjectData:
    import mne

    folder = (root or deriv_dir()) / subject
    wave = mne.read_epochs(folder / "waveform-epo.fif", preload=True, verbose="error")
    trials = pd.read_csv(folder / "trials.tsv", sep="\t")
    y = trials.label.to_numpy().astype(int)
    if not np.array_equal(wave.events[:, 2] - 1, y):
        raise RuntimeError(f"{subject}: waveform epochs and trials.tsv disagree")
    data = SubjectData(
        subject=subject,
        ch_names=list(wave.ch_names),
        y=y,
        wave=wave.get_data(copy=True) * 1e6,
        wave_times=wave.times.copy(),
        trials=trials,
    )
    if envelopes:
        for band in BAND_ORDER:
            ep = mne.read_epochs(folder / f"env-{band}-epo.fif", preload=True, verbose="error")
            if ep.ch_names != data.ch_names or len(ep) != len(y):
                raise RuntimeError(f"{subject}/{band}: envelope epochs do not match the waveform")
            data.env[band] = ep.get_data(copy=True)
            data.env_times = ep.times.copy()
    if tfr:
        with np.load(folder / "tfr-epo.npz") as f:
            if list(f["ch_names"]) != data.ch_names or len(f["coef"]) != len(y):
                raise RuntimeError(f"{subject}: tfr-epo.npz does not match the waveform")
            data.tfr = f["coef"]
            data.tfr_times = f["times"]
    return data


# --- envelope transforms ----------------------------------------------------------------------


def _baseline_mask(times: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    mask = (times >= baseline[0] - TIME_TOL) & (times <= baseline[1] + TIME_TOL)
    if not mask.any():
        raise ValueError(f"no samples in baseline {baseline}")
    return mask


def log_baseline(
    amplitude: np.ndarray, times: np.ndarray, cfg: FeatureConfig = FeatureConfig()
) -> np.ndarray:
    """``log10(A)`` minus its mean over the envelope baseline, per trial and channel."""
    x = np.log10(np.maximum(amplitude, cfg.eps))
    mask = _baseline_mask(times, cfg.env_baseline)
    return x - x[..., mask].mean(axis=-1, keepdims=True)


def relative_log_baseline(
    env: dict[str, np.ndarray], times: np.ndarray, cfg: FeatureConfig = FeatureConfig()
) -> dict[str, np.ndarray]:
    """``log10(A_band / sum of the four bands)`` minus its baseline mean, per trial and channel."""
    amps = {b: np.maximum(env[b], cfg.eps) for b in BAND_ORDER}
    total = sum(amps.values())
    mask = _baseline_mask(times, cfg.env_baseline)
    out = {}
    for b in BAND_ORDER:
        r = np.log10(amps[b] / total)
        out[b] = r - r[..., mask].mean(axis=-1, keepdims=True)
    return out


# --- windows and bins -------------------------------------------------------------------------


def bin_masks(times: np.ndarray, window: tuple[float, float], width: float) -> list[np.ndarray]:
    lo, hi = window
    n_bins = int(round((hi - lo) / width))
    if not np.isclose(n_bins * width, hi - lo):
        raise ValueError(f"window {window} is not a multiple of {width}")
    masks = []
    for k in range(n_bins):
        start, stop = lo + k * width, lo + (k + 1) * width
        mask = (times >= start - TIME_TOL) & (times < stop - TIME_TOL)
        if not mask.any():
            raise ValueError(f"empty bin [{start}, {stop}) in the available times")
        masks.append(mask)
    return masks


def bin_average(
    x: np.ndarray, times: np.ndarray, window: tuple[float, float], width: float = 0.05
) -> np.ndarray:
    """``(..., times)`` -> ``(..., bins)`` averages over consecutive bins of the window."""
    return np.stack([x[..., m].mean(axis=-1) for m in bin_masks(times, window, width)], axis=-1)


# --- channels ---------------------------------------------------------------------------------


def resolve(names: tuple[str, ...] | list[str], available: list[str]) -> list[int]:
    """Indices of ``names`` in ``available``, case-insensitively."""
    lookup = {c.lower(): i for i, c in enumerate(available)}
    missing = [n for n in names if n.lower() not in lookup]
    if missing:
        raise KeyError(f"channels not found: {missing}")
    return [lookup[n.lower()] for n in names]


def channel_matrix(code: str, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Linear map ``(virtual channels x channels)`` for channel set ``code``."""
    n = len(ch_names)
    if code == "C1":
        ref = {r.lower() for r in REFERENCE_CHANNELS}
        idx = [i for i, c in enumerate(ch_names) if c.lower() not in ref]
        names = [ch_names[i] for i in idx]
        m = np.zeros((len(idx), n))
        m[np.arange(len(idx)), idx] = 1.0
        return m, names
    if code == "C5":
        rows, names = [], []
        for left, right in SYMMETRIC_PAIRS:
            i, j = resolve((left, right), ch_names)
            row = np.zeros(n)
            row[[i, j]] = 0.5
            rows.append(row)
            names.append(f"{ch_names[i]}+{ch_names[j]}")
        for mid in MIDLINE:
            (i,) = resolve((mid,), ch_names)
            row = np.zeros(n)
            row[i] = 1.0
            rows.append(row)
            names.append(ch_names[i])
        return np.vstack(rows), names
    chosen = CHANNEL_SETS[code]
    if chosen is None:
        raise ValueError(code)
    idx = resolve(chosen, ch_names)
    m = np.zeros((len(idx), n))
    m[np.arange(len(idx)), idx] = 1.0
    return m, [ch_names[i] for i in idx]


def apply_channels(x: np.ndarray, code: str, ch_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """``(trials, channels, t)`` -> ``(trials, virtual channels, t)``."""
    m, names = channel_matrix(code, ch_names)
    return np.einsum("vc,nct->nvt", m, x), names


# --- assembly ---------------------------------------------------------------------------------


def component_series(
    data: SubjectData, component: str, cfg: FeatureConfig, cache: dict | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(trials, channels, times)`` and its time axis for one component of D1."""
    cache = {} if cache is None else cache
    if component == "wave":
        return data.wave, data.wave_times
    kind, _, k = component.partition(":")
    if kind in ("amp", "cos", "sin"):
        if data.tfr is None:
            raise RuntimeError(f"{data.subject}: time-frequency coefficients not loaded")
        if component not in cache:
            z = data.tfr[:, int(k)].astype(np.complex128)  # stored as complex64
            if kind == "amp":
                cache[component] = log_baseline(np.abs(z), data.tfr_times, cfg)
            else:
                phase = np.angle(z)
                cache[component] = np.cos(phase) if kind == "cos" else np.sin(phase)
        return cache[component], data.tfr_times
    if component.startswith("rel:"):
        if "rel" not in cache:
            cache["rel"] = relative_log_baseline(data.env, data.env_times, cfg)
        return cache["rel"][component[4:]], data.env_times
    key = f"log:{component}"
    if key not in cache:
        cache[key] = log_baseline(data.env[component], data.env_times, cfg)
    return cache[key], data.env_times


def build_features(
    data: SubjectData,
    config: Config,
    cfg: FeatureConfig = FeatureConfig(),
    cache: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """``(trials, components, channels, bins)`` for the B, W and C of ``config``."""
    window = WINDOWS[config.W]
    blocks = []
    names: list[str] = []
    for comp in components(config.B):
        series, times = component_series(data, comp, cfg, cache)
        binned = bin_average(series, times, window, cfg.bin_width)
        chans, names = apply_channels(binned, config.C, data.ch_names)
        blocks.append(chans)
    return np.stack(blocks, axis=1), names


def flatten(x4: np.ndarray) -> np.ndarray:
    return x4.reshape(len(x4), -1)
