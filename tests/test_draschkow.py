import numpy as np
import pandas as pd
import pytest

from n400fp import draschkow
from n400fp.config import PreprocConfig
from n400fp.preprocess import preproc_config


def test_marker_codes_parse_all_stimulus_formats():
    desc = ["Stimulus/201", "Stimulus/  6", "Stimulus/S111", "Comment/actiCAP Data On",
            "Response/R  1", "New Segment/"]  # fmt: skip
    assert draschkow.marker_codes(desc).tolist() == [201, 6, 111, -1, -1, -1]


def test_scene_targets():
    onsets = [1.0, 2.0, 3.0, 4.0, 5.0]
    desc = ["Stimulus/S101", "Stimulus/S112", "Stimulus/S103", "Stimulus/S111", "Stimulus/S121"]
    t = draschkow.target_table(onsets, desc, "scene")
    assert t.onset.tolist() == [2.0, 4.0]  # no display delay
    assert t.label.tolist() == [1, 0]
    assert t.condition.tolist() == ["inconsistent", "consistent"]
    assert t.event_index.tolist() == [0, 1]
    assert t.rt_ms.isna().all()


def test_sentence_targets_use_item_code_delay_and_lookup():
    lookup = pd.Series(
        [2.5, 1.0, 3.0],
        index=pd.MultiIndex.from_tuples([(201, 5), (210, 37), (200, 18)], names=["trig", "item"]),
    )
    onsets = [10.0, 10.0, 20.0, 20.5, 21.0, 30.0, 31.0]
    desc = ["Stimulus/  6", "Stimulus/201",  # item code just before the condition marker
            "Stimulus/ 38", "Comment/x", "Stimulus/210",  # non-stimulus annotation in between
            "Stimulus/ 19", "Stimulus/200"]  # fmt: skip
    t = draschkow.target_table(onsets, desc, "sentence", lookup)
    assert np.allclose(t.onset, [10.0 + 0.255 + 2.5, 21.0 + 0.255 + 1.0, 31.0 + 0.255 + 3.0])
    assert t.label.tolist() == [0, 1, 0]
    assert t.item.tolist() == [5, 37, 18]
    assert t.condition.tolist() == ["congruent", "incongruent", "congruent"]


def test_sentence_targets_fail_loudly():
    lookup = pd.Series([1.0], index=pd.MultiIndex.from_tuples([(201, 5)]))
    with pytest.raises(KeyError):
        draschkow.target_table([1.0, 2.0], ["Stimulus/  9", "Stimulus/201"], "sentence", lookup)
    with pytest.raises(ValueError):
        draschkow.target_table([1.0], ["Stimulus/201"], "sentence", lookup)


def test_keep_targets_annotations(tmp_path):
    import mne

    root = tmp_path
    (root / "csv").mkdir()
    lookup = {
        "label": ["con_hc_5", "inc_lc_7", "abst_1"],
        "trig": [201, 210, 100],
        "onset": [1.0, 2.0, 0.5],
    }
    pd.DataFrame(lookup).to_csv(root / draschkow.LOOKUP)
    draschkow._lookup.cache_clear()
    info = mne.create_info(["Cz", "SO1"], 1000.0, ["eeg", "eog"])
    raw = mne.io.RawArray(np.zeros((2, 20000)), info, verbose="error")
    markers = ["Stimulus/  6", "Stimulus/201", "Stimulus/  8", "Stimulus/210"]
    raw.set_annotations(mne.Annotations([1, 1.5, 9, 9.5], 0, markers))
    trials = draschkow.keep_targets(raw, "sentence", root)
    assert list(raw.annotations.description) == ["related", "unrelated"]
    assert np.allclose(raw.annotations.onset, [1.5 + 0.255 + 1.0, 9.5 + 0.255 + 2.0])
    assert trials.label.tolist() == [0, 1]


def test_linked_mastoid_conversion():
    import mne

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1000))
    raw = mne.io.RawArray(data.copy(), mne.create_info(["Fz", "Cz", "M1", "M2"], 1000.0, "eeg"),
                          verbose="error")  # fmt: skip
    assert draschkow.to_linked_mastoids(raw)
    assert raw.ch_names == ["Fz", "Cz"]
    assert np.allclose(raw.get_data(), data[:2] - data[2:].mean(axis=0))
    raw62 = mne.io.RawArray(data[:2].copy(), mne.create_info(["Fz", "Cz"], 1000.0, "eeg"),
                            verbose="error")  # fmt: skip
    assert not draschkow.to_linked_mastoids(raw62)
    assert np.allclose(raw62.get_data(), data[:2])


def test_subject_codes_and_paths(tmp_path):
    folder = tmp_path / "unprocessed"
    folder.mkdir()
    for code in ("ZZA01", "AHA31", "BJA19"):
        for task in ("sentence", "vis_scene"):
            (folder / f"{code}_{task}.vhdr").write_text("")
    assert draschkow.subject_codes(tmp_path) == ["AHA31", "BJA19", "ZZA01"]
    assert draschkow.raw_path("sub-002", "scene", tmp_path).name == "BJA19_vis_scene.vhdr"
    assert draschkow.raw_path("sub-003", "sentence", tmp_path).name == "ZZA01_sentence.vhdr"


def test_preproc_config_per_dataset():
    assert preproc_config("erpcore") == PreprocConfig()
    for task in ("scene", "sentence"):
        cfg = preproc_config(task, n_jobs=3)
        assert cfg.initial_reference == () and cfg.reference == () and cfg.n_jobs == 3
        assert cfg.ica_n_components == PreprocConfig().ica_n_components
        assert len(cfg.keep_channels) == 28 and not {"P9", "P10"} & set(cfg.keep_channels)
    scene, sentence = preproc_config("scene"), preproc_config("sentence")
    assert scene.muscle_ica and scene.bad_trials == "interpolate"
    assert not sentence.muscle_ica and sentence.bad_trials == "reject"
    assert preproc_config("sentence", muscle_ica=True).muscle_ica  # overrides win


def test_prespecified_variant_is_the_path_before_gate_1(monkeypatch):
    """Guide 13.11: as 13.3 (no re-referencing) without the three changes of 13.8."""
    default = PreprocConfig()
    for task in ("scene", "sentence"):
        cfg = preproc_config(task, variant="prespecified", n_jobs=2)
        assert cfg.initial_reference == () and cfg.reference == () and cfg.n_jobs == 2
        assert cfg.keep_channels == () and cfg.muscle_ica and cfg.bad_trials == "interpolate"
        assert cfg.ica_n_components == default.ica_n_components
    monkeypatch.setenv("N400_DATASET", "sentence")
    monkeypatch.setenv("N400_PREPROC", "prespecified")
    assert preproc_config("sentence") == preproc_config("sentence", variant="prespecified")
    with pytest.raises(ValueError, match="erpcore"):
        preproc_config("erpcore", variant="prespecified")
    with pytest.raises(ValueError, match="variant"):
        preproc_config("scene", variant="other")


def test_kept_channels_case_insensitive_in_recording_order():
    from n400fp.preprocess import kept_channels

    cfg = PreprocConfig(keep_channels=("FP1", "Cz", "O2"))
    assert kept_channels(["O2", "Fp1", "SO1", "Cz", "C5"], cfg) == ["O2", "Fp1", "Cz"]
    assert kept_channels(["A", "B"], PreprocConfig()) == ["A", "B"]
    with pytest.raises(ValueError, match="O2"):
        kept_channels(["Fp1", "Cz"], cfg)


def test_reject_trials_uses_threshold_on_kept_channels():
    import mne

    from n400fp.preprocess import reject_trials

    rng = np.random.default_rng(0)
    names = ["Fp1", "Cz", "Pz", "Oz", "C3", "C4"]
    data = rng.normal(0, 5e-6, (60, len(names), 128))
    data[[3, 17, 40], 1] += 400e-6 * np.sin(np.linspace(0, np.pi, 128))  # Cz artefacts
    data[[8], 0] += 400e-6  # artefact only on Fp1, which is not kept
    data[8, 0, :64] -= 400e-6
    info = mne.create_info(names, 128.0, "eeg")
    epochs = mne.EpochsArray(data, info, verbose="error")
    cfg = PreprocConfig(keep_channels=("Cz", "Pz", "Oz", "C3", "C4"))
    repaired, interp, dropped, threshold, flat = reject_trials(epochs, cfg)
    ptp_kept = np.ptp(data[:, 1:], axis=2).max(axis=1)  # Fp1 (index 0) is not kept
    assert not flat.any() and (dropped == (ptp_kept > threshold)).all()
    assert dropped[[3, 17, 40]].all() and 20e-6 < threshold < 400e-6
    assert len(repaired) == (~dropped).sum() and repaired.ch_names == names
    assert interp.shape == (len(repaired), len(names)) and not interp.any()


def test_paths_are_scoped_by_dataset(monkeypatch, tmp_path):
    from n400fp import paths

    monkeypatch.setenv("N400_DERIV_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("N400_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.delenv("N400_DATASET", raising=False)
    assert paths.deriv_dir() == tmp_path / "d" and paths.results_dir() == tmp_path / "r"
    monkeypatch.setenv("N400_DATASET", "scene")
    assert paths.deriv_dir() == tmp_path / "d" / "scene"
    assert paths.results_dir() == tmp_path / "r" / "scene"
    assert paths.task_results_dir("erpcore") == tmp_path / "r"
    monkeypatch.setenv("N400_DATASET", "other")
    with pytest.raises(ValueError):
        paths.deriv_dir()


def test_prespecified_variant_has_its_own_folders(monkeypatch, tmp_path):
    from n400fp import paths

    monkeypatch.setenv("N400_DERIV_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("N400_RESULTS_DIR", str(tmp_path / "r"))
    monkeypatch.setenv("N400_DATASET", "scene")
    monkeypatch.setenv("N400_PREPROC", "prespecified")
    assert paths.deriv_dir() == tmp_path / "d" / "preproc_prespecified" / "scene"
    assert paths.results_dir() == tmp_path / "r" / "preproc_prespecified" / "scene"
    assert paths.task_results_dir("scene") == tmp_path / "r" / "scene"  # main results
    assert paths.prespecified_results_dir("sentence") == paths.results_dir().parent / "sentence"
    with pytest.raises(ValueError):
        paths.prespecified_results_dir("erpcore")
    for dataset in ("erpcore", "pursue"):  # nothing changed after gate 1 there
        monkeypatch.setenv("N400_DATASET", dataset)
        with pytest.raises(ValueError, match="N400_PREPROC"):
            paths.results_dir()
    monkeypatch.setenv("N400_DATASET", "scene")
    monkeypatch.setenv("N400_PREPROC", "other")
    with pytest.raises(ValueError, match="N400_PREPROC"):
        paths.deriv_dir()
    monkeypatch.setenv("N400_PREPROC", "")  # empty = unset
    assert paths.deriv_dir() == tmp_path / "d" / "scene"


def test_cli_rejects_the_variant_before_running_a_command(monkeypatch):
    from n400fp import cli

    monkeypatch.setenv("N400_PREPROC", "prespecified")
    with pytest.raises(ValueError, match="N400_PREPROC"):
        cli.main(["--dataset", "erpcore", "plan", "--stage", "baseline"])


def test_reject_trials_drops_flat_trials_before_the_threshold():
    """A partly failed recording: without the flat rule the threshold falls to microvolts."""
    import mne

    from n400fp.preprocess import reject_trials

    rng = np.random.default_rng(1)
    names = ["Cz", "Pz", "Oz", "C3", "C4"]
    data = rng.normal(0, 10e-6, (80, len(names), 128))
    data[:25] *= 0.05  # flat: SD 0.5 uV
    data[[40, 41, 42, 43, 44, 45, 46, 47]] *= 30  # large artefacts
    epochs = mne.EpochsArray(data, mne.create_info(names, 128.0, "eeg"), verbose="error")
    repaired, _, dropped, threshold, flat = reject_trials(epochs, PreprocConfig())
    assert flat.tolist() == [True] * 25 + [False] * 55
    assert dropped[:25].all() and dropped[40:48].all()
    assert threshold > 20e-6 and len(repaired) >= 40
