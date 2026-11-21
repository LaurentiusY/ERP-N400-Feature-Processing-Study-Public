import json

import pytest

from n400fp.summary import channel_sets, read_preproc
from n400fp.synthetic import channel_names


def _meta(subject: str, channels: list[str]) -> dict:
    return {
        "subject": subject,
        "n_targets": 120,
        "n_kept": 120,
        "n_related": 60,
        "n_unrelated": 60,
        "n_dropped_epoching": 0,
        "n_dropped_autoreject": 0,
        "ica_eog": {"n_components": 2, "explained_variance": 0.3},
        "ica_muscle": {"n_components": 1, "explained_variance": 0.02},
        "interpolated_fraction": 0.05,
        "channels": channels,
        "behaviour": {"n_no_response": 1, "median_rt_related": 550.0},
    }


def test_read_preproc_skips_missing_subjects(tmp_path):
    names = channel_names()
    for subject in ("sub-001", "sub-003"):
        (tmp_path / subject).mkdir()
        (tmp_path / subject / "preproc.json").write_text(json.dumps(_meta(subject, names)))
    table, channels = read_preproc(["sub-001", "sub-002", "sub-003"], tmp_path)
    assert table.subject.tolist() == ["sub-001", "sub-003"]
    assert table.n_channels.tolist() == [30, 30]
    assert table.ica_muscle_n.tolist() == [1, 1]
    assert table.median_rt_related.tolist() == [550.0, 550.0]
    assert table.median_rt_unrelated.isna().all()  # missing behaviour fields stay empty
    assert set(channels) == {"sub-001", "sub-003"}


def test_read_preproc_without_any_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_preproc(["sub-001"], tmp_path)


def test_channel_sets_on_recorded_names():
    sets = channel_sets(channel_names())  # "Fp1"/"Fp2" as after preprocessing
    assert [len(sets[c]) for c in ("C1", "C2", "C3", "C4", "C5")] == [28, 7, 7, 7, 17]
    assert not {"P9", "P10"} & set(sets["C1"])
    assert "Fp1+Fp2" in sets["C5"]


def test_channel_sets_report_missing_channels():
    names = [c for c in channel_names() if c != "CPz"]
    with pytest.raises(KeyError):
        channel_sets(names)


def test_channel_sets_may_leave_out_unavailable_levels():
    names = [c for c in channel_names() if c != "FCz"]  # PURSUE
    sets = channel_sets(names, allow_missing=True)
    assert sets["C5"] == [] and len(sets["C2"]) == 7
