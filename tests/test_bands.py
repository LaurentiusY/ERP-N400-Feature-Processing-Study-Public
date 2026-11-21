import numpy as np
import pytest
from scipy.signal import hilbert

from n400fp.bands import interpolate_per_trial, interpolated_envelope
from n400fp.preprocess import set_montage, unrepairable
from n400fp.synthetic import channel_names


def _epochs(data):
    import mne

    info = mne.create_info(channel_names(), 256.0, "eeg")
    n = len(data)
    events = np.column_stack([np.arange(n) * 1000, np.zeros(n, int), np.ones(n, int)])
    epochs = mne.EpochsArray(data, info, events=events, tmin=-1.0, baseline=None, verbose="error")
    set_montage(epochs)
    return epochs


@pytest.fixture
def band_data():
    rng = np.random.default_rng(0)
    n_trials, n_ch, n_times = 6, 30, 400
    t = np.arange(n_times) / 256.0
    # alpha-like oscillations with spatially smooth amplitude, plus noise
    phase = rng.uniform(0, 2 * np.pi, (n_trials, n_ch, 1))
    x = np.sin(2 * np.pi * 10 * t + phase) * rng.uniform(1, 3, (n_trials, n_ch, 1)) * 1e-6
    x += 0.2e-6 * rng.standard_normal(x.shape)
    interp = np.zeros((n_trials, n_ch), bool)
    interp[1, [4, 13, 14]] = True  # C3, Pz, CPz
    interp[3, :25] = True  # most channels, as in heavily repaired trials
    return x, interp


def test_envelope_equals_envelope_of_interpolated_voltage(band_data):
    x, interp = band_data
    ours = interpolated_envelope(_epochs(hilbert(x, axis=-1)), interp).get_data()
    voltage = interpolate_per_trial(_epochs(x), interp).get_data()
    reference = np.abs(hilbert(voltage, axis=-1))
    assert np.allclose(ours, reference, rtol=1e-8, atol=1e-18)
    assert (ours >= 0).all()
    # untouched trials keep their envelope
    assert np.allclose(ours[0], np.abs(hilbert(x[0], axis=-1)))


def test_interpolating_the_envelope_itself_can_go_negative(band_data):
    x, interp = band_data
    env = np.abs(hilbert(x, axis=-1))
    assert (interpolate_per_trial(_epochs(env), interp).get_data() < 0).any()


def test_unrepairable_trials():
    interp = np.zeros((4, 30), bool)
    interp[1] = True
    interp[2, :29] = True  # one good channel left: still interpolated, not dropped
    assert unrepairable(interp).tolist() == [False, True, False, False]
