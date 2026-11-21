"""End-to-end run on a synthetic ERP CORE-like recording (ICA, autoreject, envelopes).

Run with ``pytest -m slow`` (about one to a few minutes).
"""

import json

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    from n400fp.bands import compute_envelopes
    from n400fp.config import PreprocConfig
    from n400fp.preprocess import preprocess_raw
    from n400fp.synthetic import make_raw

    root = tmp_path_factory.mktemp("deriv")
    raw = make_raw(n_trials=60, effect_uv=10.0, seed=3)
    cfg = PreprocConfig(n_jobs=1)
    meta = preprocess_raw(raw, "sub-001", root / "sub-001", cfg)
    compute_envelopes("sub-001", cfg, deriv_root=root)
    return root, meta


def test_preprocessing_outputs(processed):
    root, meta = processed
    folder = root / "sub-001"
    for name in ("clean-raw.fif", "waveform-epo.fif", "interp_labels.npy", "trials.tsv",
                 "preproc.json"):  # fmt: skip
        assert (folder / name).exists(), name
    saved = json.loads((folder / "preproc.json").read_text())
    assert saved["n_targets"] == 60
    assert saved["n_related"] + saved["n_unrelated"] == saved["n_kept"]
    assert len(saved["channels"]) == 30
    trials = pd.read_csv(folder / "trials.tsv", sep="\t")
    assert len(trials) == saved["n_kept"]
    assert trials.rt_ms.notna().all()
    # related responses were simulated faster than unrelated ones
    med = trials.groupby("condition").rt_ms.median()
    assert med["related"] < med["unrelated"]
    interp = np.load(folder / "interp_labels.npy")
    assert interp.shape == (saved["n_kept"], 30) and interp.dtype == bool


def test_epochs_and_envelopes_align(processed):
    import mne

    from n400fp.config import BAND_ORDER

    root, meta = processed
    folder = root / "sub-001"
    wave = mne.read_epochs(folder / "waveform-epo.fif", verbose="error")
    assert wave.info["sfreq"] == 256
    assert np.isclose(wave.times[0], -0.3984375) and wave.times[-1] >= 0.8 - 1e-9
    for band in BAND_ORDER:
        env = mne.read_epochs(folder / f"env-{band}-epo.fif", verbose="error")
        assert env.ch_names == wave.ch_names
        assert np.array_equal(env.events, wave.events)
        assert np.isclose(env.times[0], -0.5) and np.isclose(env.times[-1], wave.times[-1])
        shared = np.intersect1d(np.round(env.times * 256), np.round(wave.times * 256))
        assert len(shared) == len(wave.times)  # same sample grid
        assert (env.get_data() >= 0).all()  # interpolated envelopes stay non-negative


def test_tfr_coefficients_align_with_the_waveform(processed):
    """Guide 13.10: Morlet coefficients on the continuous data, cut like the envelopes."""
    import mne

    from n400fp.config import TFR_CYCLES, TFR_FREQS, DecodeConfig, PreprocConfig
    from n400fp.features import load_subject
    from n400fp.preprocess import EVENT_ID
    from n400fp.run_subject import run_subject, subject_means
    from n400fp.study import Config
    from n400fp.tfr import compute_tfr, morlet_segment, tfr_file

    root, meta = processed
    folder = root / "sub-001"
    compute_tfr("sub-001", PreprocConfig(n_jobs=1), deriv_root=root)
    wave = mne.read_epochs(folder / "waveform-epo.fif", verbose="error")
    with np.load(tfr_file("sub-001", root)) as f:
        coef, times, names = f["coef"], f["times"], list(f["ch_names"])
    assert coef.shape == (len(wave), len(TFR_FREQS), len(wave.ch_names), len(times))
    assert coef.dtype == np.complex64 and names == wave.ch_names
    assert not (folder / "tfr-epo.tmp.npz").exists()  # written to a temporary name, then renamed
    assert np.isclose(times[0], -0.5) and np.isclose(times[-1], wave.times[-1])

    # channels not interpolated in a trial equal the wavelet transform of the raw data
    raw = mne.io.read_raw_fif(folder / "clean-raw.fif", preload=True, verbose="error")
    trials = pd.read_csv(folder / "trials.tsv", sep="\t")
    interp = np.load(folder / "interp_labels.npy")
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose="error")
    k = 4
    direct = morlet_segment(raw.get_data(), 256.0, TFR_FREQS[k], TFR_CYCLES[k])
    for i in (0, len(trials) - 1):
        good = ~interp[i]
        first = events[trials.event_index[i], 0] - raw.first_samp + int(round(-0.5 * 256))
        expected = direct[good, first : first + len(times)]
        assert np.allclose(coef[i, k, good], expected, rtol=1e-4, atol=1e-12)
    assert interp.any()  # and the interpolated ones were replaced
    i, j = np.argwhere(interp)[0]
    first = events[trials.event_index[i], 0] - raw.first_samp + int(round(-0.5 * 256))
    assert not np.allclose(coef[i, k, j], direct[j, first : first + len(times)], rtol=1e-4)

    data = load_subject("sub-001", root, envelopes=False, tfr=True)
    fast = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))
    base = [Config.parse("B1-W1-N2-C1")]
    run_subject("sub-001", "baseline", ("lr",), base, results_root=root / "main", dcfg=fast,
                data=data)  # fmt: skip
    df = run_subject("sub-001", "tfr", ("lr",), results_root=root / "main" / "tfr_sensitivity",
                     dcfg=fast, data=data, splits_root=root / "main")  # fmt: skip
    means = subject_means(df).set_index("B").bacc
    assert set(means.index) == {"T1", "T2", "T3", "T4", "T5"}
    assert means["T5"] > 0.6


def test_target_close_to_the_end_of_the_recording(tmp_path):
    from n400fp.bands import compute_envelopes
    from n400fp.config import PreprocConfig
    from n400fp.features import load_subject
    from n400fp.preprocess import preprocess_raw
    from n400fp.synthetic import make_raw

    raw = make_raw(n_trials=20, seed=7)
    last_target = max(
        a["onset"] for a in raw.annotations if a["description"].startswith(("21", "22"))
    )
    raw.crop(tmax=last_target + 1.0)  # room for the waveform epoch (0.8 s), not much more
    cfg = PreprocConfig(n_jobs=1)
    meta = preprocess_raw(raw, "sub-001", tmp_path / "sub-001", cfg)
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)
    assert meta["n_targets"] == 20
    assert all(e.shape[0] == meta["n_kept"] for e in data.env.values())


def draschkow_like_raw(seed: int = 3):
    """Synthetic recording in the Draschkow format: Stimulus/S111-112 markers, SO1/SO2, M1/M2."""
    import mne

    from n400fp.synthetic import make_raw

    raw = make_raw(n_trials=60, effect_uv=10.0, seed=seed)
    mastoids = raw.copy().pick(["P9", "P10"]).rename_channels({"P9": "M1", "P10": "M2"})
    raw.add_channels([mastoids], force_update_info=True)
    raw.drop_channels(["HEOG_right"])
    raw.rename_channels({"HEOG_left": "SO1", "VEOG_lower": "SO2"})
    raw.set_channel_types({"SO1": "eog", "SO2": "eog"})
    marker = {"21": "S111", "22": "S112"}
    raw.set_annotations(
        mne.Annotations(
            raw.annotations.onset,
            0,
            [f"Stimulus/{marker.get(d[:2], 'S101')}" for d in raw.annotations.description],
        )
    )
    return raw


def test_draschkow_scene_path_end_to_end(tmp_path):
    from n400fp.bands import compute_envelopes
    from n400fp.config import ERPCORE_ELECTRODES, DecodeConfig
    from n400fp.features import load_subject
    from n400fp.preprocess import preproc_config, preprocess_raw
    from n400fp.run_subject import run_subject, subject_means
    from n400fp.study import Config

    raw = draschkow_like_raw()
    cfg = preproc_config("scene", n_jobs=1)
    meta = preprocess_raw(raw, "sub-001", tmp_path / "sub-001", cfg, "scene")
    assert meta["dataset"] == "scene" and meta["converted_to_linked_mastoids"]
    assert meta["n_targets"] == 60 and meta["behaviour"] == {}
    assert sorted(c.upper() for c in meta["channels"]) == sorted(
        c.upper() for c in ERPCORE_ELECTRODES
    )
    assert len(meta["interpolated_fraction_by_channel"]) == 30  # P9/P10 kept for ICA/interpolation
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)  # checks that envelope and waveform channels match
    assert all(e.shape[0] == meta["n_kept"] for e in data.env.values())
    fast = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))
    df = run_subject("sub-001", "baseline", models=("lr",), configs=[Config.parse("B1-W1-N2-C1")],
                     results_root=tmp_path / "results", dcfg=fast, data=data)  # fmt: skip
    assert subject_means(df).bacc.iloc[0] > 0.6


def test_trial_without_good_channel_is_dropped(tmp_path, monkeypatch):
    import mne

    from n400fp import preprocess
    from n400fp.bands import compute_envelopes
    from n400fp.config import PreprocConfig
    from n400fp.features import load_subject
    from n400fp.synthetic import make_raw

    real = preprocess.autoreject_interpolate

    def flag_every_channel_of_trial_5(epochs, cfg):
        repaired, interp, dropped = real(epochs, cfg)
        interp = interp.copy()
        interp[5] = True
        return repaired, interp, dropped

    monkeypatch.setattr(preprocess, "autoreject_interpolate", flag_every_channel_of_trial_5)
    cfg = PreprocConfig(n_jobs=1)
    raw = make_raw(n_trials=20, seed=5)
    meta = preprocess.preprocess_raw(raw, "sub-001", tmp_path / "sub-001", cfg)
    assert meta["n_dropped_unrepairable"] == 1 and meta["unrepairable_targets"] == [5]
    assert meta["n_kept"] == 19
    trials = pd.read_csv(tmp_path / "sub-001" / "trials.tsv", sep="\t")
    assert 5 not in trials.event_index.tolist()
    wave = mne.read_epochs(tmp_path / "sub-001" / "waveform-epo.fif", verbose="error")
    assert len(wave) == 19 and not (wave.get_data() == 0).all(axis=(1, 2)).any()
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)  # checks labels, channels and trial counts
    assert all(e.shape[0] == 19 for e in data.env.values())


def test_features_and_baseline_decoding(processed):
    from n400fp.config import DecodeConfig
    from n400fp.features import build_features, load_subject
    from n400fp.run_subject import run_subject, subject_means
    from n400fp.study import full_factorial, pilot

    root, _ = processed
    data = load_subject("sub-001", root)
    assert data.wave.std() > 0.5  # microvolts, not volts
    for cfg in full_factorial()[::37]:
        x, _ = build_features(data, cfg)
        assert np.isfinite(x).all()
    fast = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))
    df = run_subject("sub-001", "pilot", configs=pilot()[:2], results_root=root / "results",
                     dcfg=fast, data=data)  # fmt: skip
    means = subject_means(df)
    base = means[(means.model == "lr") & (means.config == "B1-W1-N2-C1")].bacc.iloc[0]
    assert base > 0.6


def test_draschkow_sentence_settings_end_to_end(tmp_path):
    """Sentence settings: no muscle ICA, trials above the threshold dropped, none interpolated."""
    from n400fp.bands import compute_envelopes
    from n400fp.features import load_subject
    from n400fp.preprocess import preproc_config, preprocess_raw

    raw = draschkow_like_raw(seed=4)
    cfg = preproc_config("sentence", n_jobs=1)
    meta = preprocess_raw(raw, "sub-001", tmp_path / "sub-001", cfg, "scene")  # scene markers
    assert meta["ica_muscle"]["n_components"] == 0 and meta["interpolated_fraction"] == 0.0
    assert meta["n_dropped_autoreject"] == 0 and meta["rejection_threshold_uv"] > 0
    assert meta["n_kept"] + meta["n_dropped_threshold"] + meta["n_dropped_flat"] == 60
    assert len(meta["channels"]) == 28
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)
    assert data.wave.shape[:2] == (meta["n_kept"], 28)
    assert all(e.shape[:2] == (meta["n_kept"], 28) for e in data.env.values())


def test_pursue_path_end_to_end(tmp_path):
    """Synthetic recording in the PURSUE layout: TP9/TP10, no FCz, HEOG R/L and VEOG L."""
    from n400fp.bands import compute_envelopes
    from n400fp.config import DecodeConfig
    from n400fp.features import load_subject
    from n400fp.preprocess import preproc_config, preprocess_raw
    from n400fp.pursue import ELECTRODES
    from n400fp.run_subject import run_subject, subject_means
    from n400fp.study import stage_configs
    from n400fp.synthetic import make_raw

    raw = make_raw(n_trials=60, effect_uv=10.0, seed=5)
    raw.drop_channels(["FCz"])
    raw.rename_channels({"P9": "TP9", "P10": "TP10", "HEOG_left": "HEOG L",
                         "HEOG_right": "HEOG R", "VEOG_lower": "VEOG L"})  # fmt: skip
    cfg = preproc_config("pursue", n_jobs=1)
    meta = preprocess_raw(
        raw, "sub-001", tmp_path / "sub-001", cfg, "pursue", {"subject_code": "1001"}
    )
    assert meta["n_targets"] == 60 and meta["behaviour"]  # responses are kept, as for ERP CORE
    assert sorted(c.upper() for c in meta["channels"]) == sorted(c.upper() for c in ELECTRODES)
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)
    assert data.wave.shape[1] == 27 and all(e.shape[1] == 27 for e in data.env.values())
    fast = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))
    configs = stage_configs("replication")
    df = run_subject("sub-001", "replication", models=("lr",), configs=configs,
                     results_root=tmp_path / "results", dcfg=fast, data=data)  # fmt: skip
    means = subject_means(df).set_index("config").bacc
    assert means["B1-W1-N2-C1"] > 0.6


def test_target_too_close_to_the_recording_start_is_dropped_everywhere(tmp_path):
    """Waveform window (-0.4 s) inside the recording, envelope window (-0.5 s) not."""
    import mne

    from n400fp.bands import compute_envelopes
    from n400fp.config import PreprocConfig
    from n400fp.features import load_subject
    from n400fp.preprocess import preprocess_raw
    from n400fp.synthetic import make_raw

    raw = make_raw(n_trials=60, seed=7)
    ann = raw.annotations
    first = next(i for i, d in enumerate(ann.description) if d[:2] in ("21", "22"))
    onsets = ann.onset.copy()
    onsets[first] = 0.45 - 0.026  # trigger 0.45 s after the start once the LCD delay is added
    keep = [i for i in range(len(onsets)) if i == first or onsets[i] > 0.45]
    raw.set_annotations(mne.Annotations(onsets[keep], ann.duration[keep], ann.description[keep]))
    cfg = PreprocConfig(n_jobs=1)
    meta = preprocess_raw(raw, "sub-001", tmp_path / "sub-001", cfg)
    assert meta["n_dropped_envelope_window"] == 1 and meta["n_dropped_epoching"] >= 1
    trials = pd.read_csv(tmp_path / "sub-001" / "trials.tsv", sep="\t")
    assert 0 not in trials.event_index.tolist()
    compute_envelopes("sub-001", cfg, deriv_root=tmp_path)
    data = load_subject("sub-001", tmp_path)  # waveform, envelopes and trials aligned
    assert all(e.shape[0] == meta["n_kept"] for e in data.env.values())
