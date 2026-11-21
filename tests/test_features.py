import numpy as np
import pytest

from n400fp.config import FeatureConfig
from n400fp.features import (
    apply_channels,
    bin_average,
    bin_masks,
    build_features,
    channel_matrix,
    log_baseline,
    relative_log_baseline,
    resolve,
)
from n400fp.study import REPRESENTATIONS, WINDOWS, Config

N_BINS = {"W1": 12, "W2": 4, "W3": 6, "W4": 16}
N_CHANNELS = {"C1": 28, "C2": 7, "C3": 7, "C4": 7, "C5": 17}


@pytest.mark.parametrize("b", list(REPRESENTATIONS))
@pytest.mark.parametrize("w", list(WINDOWS))
@pytest.mark.parametrize("c", list(N_CHANNELS))
def test_shapes(subject, b, w, c):
    x, names = build_features(subject, Config(b, w, "N1", c))
    assert x.shape == (120, len(REPRESENTATIONS[b]), N_CHANNELS[c], N_BINS[w])
    assert len(names) == N_CHANNELS[c]
    assert np.isfinite(x).all()


def test_bins_cover_window_without_overlap(subject):
    masks = bin_masks(subject.wave_times, (0.2, 0.8), 0.05)
    stacked = np.vstack(masks)
    assert (stacked.sum(axis=0) <= 1).all()
    covered = subject.wave_times[stacked.any(axis=0)]
    assert covered.min() >= 0.2 - 1e-9 and covered.max() < 0.8
    assert all(m.sum() in (12, 13) for m in masks)  # 50 ms at 256 Hz


def test_bin_average_matches_manual(subject):
    out = bin_average(subject.wave, subject.wave_times, (0.3, 0.5), 0.05)
    m0 = bin_masks(subject.wave_times, (0.3, 0.5), 0.05)[0]
    assert np.allclose(out[..., 0], subject.wave[..., m0].mean(axis=-1))


def test_log_baseline_zero_mean_in_baseline(subject):
    cfg = FeatureConfig()
    x = log_baseline(subject.env["theta"], subject.env_times, cfg)
    mask = (subject.env_times >= -0.5 - 1e-6) & (subject.env_times <= -0.2 + 1e-6)
    assert np.allclose(x[..., mask].mean(axis=-1), 0.0, atol=1e-12)


def test_relative_amplitude_shares_sum_to_one(subject):
    env = subject.env
    total = sum(env.values())
    shares = sum(env[b] / total for b in env)
    assert np.allclose(shares, 1.0)
    rel = relative_log_baseline(env, subject.env_times)
    mask = (subject.env_times >= -0.5 - 1e-6) & (subject.env_times <= -0.2 + 1e-6)
    for b in rel:
        assert np.allclose(rel[b][..., mask].mean(axis=-1), 0.0, atol=1e-12)


def test_envelope_transform_uses_only_the_trial_itself(subject):
    """Changing one trial must not change any other trial's features."""
    env = {k: v.copy() for k, v in subject.env.items()}
    env["theta"][0] *= 50.0
    a = log_baseline(subject.env["theta"], subject.env_times)
    b = log_baseline(env["theta"], subject.env_times)
    assert np.allclose(a[1:], b[1:])
    ra = relative_log_baseline(subject.env, subject.env_times)
    rb = relative_log_baseline(env, subject.env_times)
    assert np.allclose(ra["alpha"][1:], rb["alpha"][1:])


def test_log_floor_handles_zero_or_negative_amplitude(subject):
    env = subject.env["theta"].copy()
    env[0, 0, 10] = -1e-9  # interpolation can produce tiny negative values
    assert np.isfinite(log_baseline(env, subject.env_times)).all()


def test_symmetrization_removes_antisymmetric_signal(subject):
    names = subject.ch_names
    m, virtual = channel_matrix("C5", names)
    assert m.shape == (17, 30)
    x = np.zeros((2, 30, 5))
    sym = np.zeros((2, 30, 5))
    from n400fp.study import MIDLINE, SYMMETRIC_PAIRS

    for left, right in SYMMETRIC_PAIRS:
        i, j = resolve((left, right), names)
        x[:, i], x[:, j] = 1.0, -1.0  # lateralized
        sym[:, i], sym[:, j] = 2.0, 2.0  # bilateral
    for mid in MIDLINE:
        (k,) = resolve((mid,), names)
        sym[:, k] = 3.0
    out, _ = apply_channels(x, "C5", names)
    assert np.allclose(out, 0.0)
    out_sym, _ = apply_channels(sym, "C5", names)
    assert np.allclose(out_sym[:, :11], 2.0) and np.allclose(out_sym[:, 11:], 3.0)


def test_channel_sets_and_case_insensitive_names(subject):
    _, c1 = channel_matrix("C1", subject.ch_names)
    assert "P9" not in c1 and "P10" not in c1 and len(c1) == 28
    _, c3 = channel_matrix("C3", subject.ch_names)
    assert {"Fp1", "Fp2", "FCz"}.isdisjoint(c3)
    upper = [c.upper() for c in subject.ch_names]
    assert resolve(("cz", "CPZ"), upper) == resolve(("Cz", "CPz"), subject.ch_names)
    with pytest.raises(KeyError):
        resolve(("T7",), subject.ch_names)


def test_b8_is_waveform_plus_four_envelopes(subject):
    b8, _ = build_features(subject, Config("B8", "W2", "N1", "C2"))
    b1, _ = build_features(subject, Config("B1", "W2", "N1", "C2"))
    b6, _ = build_features(subject, Config("B6", "W2", "N1", "C2"))
    assert np.allclose(b8[:, :1], b1) and np.allclose(b8[:, 1:], b6)
