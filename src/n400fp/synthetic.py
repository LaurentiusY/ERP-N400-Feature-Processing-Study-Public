"""Synthetic data for tests and local smoke runs (no real recordings needed).

* :func:`make_subject` - a :class:`SubjectData` with an N400-like negativity (unrelated
  targets, 300-500 ms, centro-parietal) and a theta-envelope increase, on spatially
  correlated noise.
* :func:`make_raw` - a continuous recording shaped like an ERP CORE N400 file as read from
  EEGLAB (30 scalp + 3 EOG channels, all typed EEG; prime, target and response triggers),
  for the end-to-end preprocessing test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BAND_ORDER, EEG_CHANNELS, EOG_CHANNELS, TFR_FREQS, TFR_LOW_MAX
from .features import SubjectData

N400_CHANNELS = {"cz", "cpz", "pz", "c3", "c4", "p3", "p4"}
NEAR_CHANNELS = {"fcz", "fc3", "fc4", "po3", "po4", "c5", "c6", "oz"}


def channel_names() -> list[str]:
    return [c.capitalize() if c in ("FP1", "FP2") else c for c in EEG_CHANNELS]


def spatial_weights(names: list[str]) -> np.ndarray:
    return np.array(
        [
            1.0 if c.lower() in N400_CHANNELS else 0.5 if c.lower() in NEAR_CHANNELS else 0.0
            for c in names
        ]
    )


def _smooth_noise(rng, shape, width: int) -> np.ndarray:
    x = rng.standard_normal(shape)
    kernel = np.ones(width) / np.sqrt(width)
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), -1, x)


def make_subject(
    n_trials: int = 120,
    effect_uv: float = 3.0,
    theta_gain: float = 0.3,
    noise_uv: float = 5.0,
    seed: int = 0,
    subject: str = "sub-001",
    sfreq: float = 256.0,
    tfr: bool = False,
) -> SubjectData:
    """``tfr=True`` adds complex time-frequency coefficients (see :func:`synthetic_tfr`)."""
    rng = np.random.default_rng(seed)
    names = channel_names()
    n_ch = len(names)
    y = rng.permutation(np.repeat([0, 1], n_trials // 2))
    wave_times = np.arange(-102, 206) / sfreq
    env_times = np.arange(-128, 206) / sfreq
    w = spatial_weights(names)

    mixing = rng.standard_normal((n_ch, 6))
    sources = _smooth_noise(rng, (n_trials, 6, len(wave_times)), 8)
    wave = np.einsum("cs,nst->nct", mixing, sources) * noise_uv / 3
    wave += rng.standard_normal(wave.shape) * noise_uv / 3
    bump = np.exp(-0.5 * ((wave_times - 0.4) / 0.08) ** 2)
    wave -= effect_uv * y[:, None, None] * w[None, :, None] * bump[None, None, :]

    env = {}
    bump_env = np.exp(-0.5 * ((env_times - 0.45) / 0.1) ** 2)
    for k, band in enumerate(BAND_ORDER):
        level = 1e-6 * (4 - k)  # more power at low frequencies
        log_noise = 0.2 * _smooth_noise(rng, (n_trials, n_ch, len(env_times)), 16)
        amp = level * np.exp(log_noise)
        if band == "theta":
            amp *= 1 + theta_gain * y[:, None, None] * w[None, :, None] * bump_env[None, None, :]
        env[band] = amp

    trials = pd.DataFrame(
        {
            "epoch_index": np.arange(n_trials),
            "label": y,
            "condition": np.where(y == 1, "unrelated", "related"),
            "rt_ms": np.where(y == 1, 720.0, 550.0) + rng.normal(0, 80, n_trials),
            "correct": 1.0,
        }
    )
    data = SubjectData(
        subject=subject,
        ch_names=names,
        y=y,
        wave=wave,
        wave_times=wave_times,
        env=env,
        env_times=env_times,
        trials=trials,
    )
    if tfr:
        data.tfr, data.tfr_times = synthetic_tfr(y, w, env_times, theta_gain, rng)
    return data


def synthetic_tfr(y, w, times, gain: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """``(trials, frequencies, channels, times)`` complex coefficients: smooth log-normal
    amplitudes (an increase after 300 ms for ``y = 1`` at the low frequencies and the
    centro-parietal channels) and random phases with a stimulus-locked part."""
    n_trials, n_ch = len(y), len(w)
    out = np.empty((n_trials, len(TFR_FREQS), n_ch, len(times)), np.complex64)
    bump = np.exp(-0.5 * ((times - 0.45) / 0.1) ** 2)
    for k, freq in enumerate(TFR_FREQS):
        amp = 1e-6 / freq * np.exp(0.2 * _smooth_noise(rng, (n_trials, n_ch, len(times)), 16))
        if freq <= TFR_LOW_MAX:
            amp *= 1 + gain * y[:, None, None] * w[None, :, None] * bump[None, None, :]
        jitter = rng.uniform(-np.pi, np.pi, (n_trials, n_ch, 1))
        phase = 2 * np.pi * freq * times[None, None, :] + 0.5 * jitter
        out[:, k] = amp * np.exp(1j * phase)
    return out, times.copy()


def _colored_noise(rng, shape, sfreq: float, exponent: float, fmin: float = 0.5) -> np.ndarray:
    """Unit-variance noise with power spectral density ~ 1/f**exponent (flat below ``fmin``)."""
    n = shape[-1]
    spec = np.fft.rfft(rng.standard_normal(shape), axis=-1)
    spec *= np.maximum(np.fft.rfftfreq(n, 1 / sfreq), fmin) ** (-exponent / 2)
    x = np.fft.irfft(spec, n=n, axis=-1)
    return x / x.std(axis=-1, keepdims=True)


def make_raw(n_trials: int = 60, sfreq: float = 512.0, effect_uv: float = 6.0, seed: int = 0):
    """An ERP CORE-like N400 recording (``mne.io.RawArray``, volts).

    The background is spatially mixed 1/f-like noise plus weaker white sensor noise. A flat
    (white) background would have the spectral slope of muscle activity, and
    ``ICA.find_bads_muscle`` would then remove every component, the N400 one included.
    """
    import mne

    rng = np.random.default_rng(seed)
    names = list(EEG_CHANNELS) + list(EOG_CHANNELS)
    labels = rng.permutation(np.repeat([0, 1], n_trials // 2))
    onsets, codes = [], []
    t = 2.0
    for i, lab in enumerate(labels):
        word = "1" if i % 2 == 0 else "2"
        prime_code = ("11" if lab == 0 else "12") + word
        target_code = ("21" if lab == 0 else "22") + word
        onsets.append(t)
        codes.append(prime_code)
        target = t + 0.2 + rng.uniform(0.9, 1.1)
        onsets.append(target)
        codes.append(target_code)
        onsets.append(target + (0.55 if lab == 0 else 0.72) + rng.normal(0, 0.05))
        codes.append("201" if rng.random() > 0.05 else "202")
        t = target + 0.2 + rng.uniform(1.4, 1.6)
    duration = t + 2.0
    n_times = int(duration * sfreq)
    times = np.arange(n_times) / sfreq

    n_sources = 12
    mixing = rng.standard_normal((len(names), n_sources))
    data = 2e-6 * mixing @ _colored_noise(rng, (n_sources, n_times), sfreq, exponent=1.5)
    data += 1e-6 * rng.standard_normal(data.shape)
    data += 3e-6 * np.sin(2 * np.pi * 10 * times)[None, :] * rng.uniform(0.5, 1.5, (len(names), 1))
    # blinks on VEOG_lower (inverted) and the frontopolar channels
    blink = np.zeros(n_times)
    for b in rng.uniform(1, duration - 1, int(duration / 4)):
        blink += np.exp(-0.5 * ((times - b) / 0.08) ** 2)
    idx = {n.lower(): i for i, n in enumerate(names)}
    for name, gain in (("fp1", 1.0), ("fp2", 1.0), ("f3", 0.3), ("f4", 0.3), ("veog_lower", -1.0)):
        data[idx[name]] += gain * 100e-6 * blink
    # the N400 generator has its own ongoing 1/f activity, like any cortical source; on its own
    # the sparse effect would form an ICA component with a flat high-frequency spectrum
    w = spatial_weights(names)
    data += 2e-6 * w[:, None] * _colored_noise(rng, (1, n_times), sfreq, exponent=1.5)
    for onset, code in zip(onsets, codes, strict=True):
        if code.startswith("22"):
            bump = np.exp(-0.5 * ((times - onset - 0.4) / 0.08) ** 2)
            data -= effect_uv * 1e-6 * w[:, None] * bump[None, :]

    info = mne.create_info(names, sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), codes))
    return raw
