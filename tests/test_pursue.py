"""PURSUE replication (guide section 14): dataset mapping, preprocessing settings, analysis."""

import hashlib
import importlib.util
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n400fp import pursue
from n400fp.events import target_trials
from n400fp.preprocess import preproc_config

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bids_root(tmp_path, codes):
    lines = ["participant_id\tAge"] + [f"sub-{c}\t20" for c in codes]
    (tmp_path / "participants.tsv").write_text("\n".join(lines) + "\n")
    return tmp_path


def all_codes():
    codes = [f"1{i:03d}" for i in range(1, 105)] + [f"2{i:03d}" for i in range(1, 79)]
    return codes + [f"3{i:03d}" for i in range(1, 107)]


def test_subject_mapping_follows_participants_tsv(tmp_path):
    root = bids_root(tmp_path, list(reversed(all_codes())))
    assert pursue.subject_code("sub-001", root) == "1001"
    assert pursue.subject_code("sub-288", root) == "3106"
    assert pursue.excluded_a_priori("sub-106", root)  # 2002
    assert not pursue.excluded_a_priori("sub-001", root)
    assert pursue.site("3042") == pursue.SITES["3"]
    path = pursue.raw_path("sub-002", root)
    assert path == root / "sub-1002" / "eeg" / "sub-1002_task-WordPR_eeg.set"


def test_incomplete_participants_table_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="288"):
        pursue.subject_codes(bids_root(tmp_path, all_codes()[:-1]))
    with pytest.raises(FileNotFoundError):
        pursue.subject_codes(tmp_path / "missing")


def test_preregistered_preprocessing_settings():
    cfg = preproc_config("pursue")
    assert cfg.initial_reference == ("Cz",) and cfg.reference == ("TP9", "TP10")
    assert cfg.muscle_ica and cfg.bad_trials == "interpolate"
    assert len(cfg.keep_channels) == 27 and "FCz" not in cfg.keep_channels
    assert not {"TP9", "TP10", "P9", "P10"} & set(cfg.keep_channels)
    assert pursue.DISPLAY_DELAY == 0.0


def test_dataset_registered_with_its_size(monkeypatch):
    from n400fp import paths

    assert paths.n_subjects("pursue") == 288 and paths.n_subjects("erpcore") == 40
    monkeypatch.setenv("N400_DATASET", "pursue")
    assert paths.subject_ids()[-1] == "sub-288"
    assert paths.deriv_dir().name == "pursue"


def test_no_display_delay_and_practice_trials_are_not_targets():
    onsets = [1.0, 2.0, 2.5, 5.0, 6.0, 6.7]
    codes = ["210", "201", "111", "211", "201", "221"]
    t = target_trials(onsets, codes, delay=0.0)
    assert t.trigger.tolist() == ["211", "221"]
    assert t.rt_ms.iloc[0] == pytest.approx(1000.0)
    assert target_trials(onsets, codes).rt_ms.iloc[0] == pytest.approx(974.0)  # ERP CORE


def test_bipolar_eog_channels():
    import mne

    names = ["FP1", "FP2", "Cz", "HEOG R", "HEOG L", "VEOG L"]
    data = np.arange(len(names))[:, None] * np.ones((1, 100)) * 1e-6
    raw = mne.io.RawArray(data, mne.create_info(names, 100.0, "eeg"), verbose="error")
    pursue.recalculate_eog(raw)
    assert raw.ch_names == ["FP1", "FP2", "Cz", "HEOG", "VEOG"]
    assert raw.get_channel_types(picks=["HEOG", "VEOG"]) == ["eog", "eog"]
    heog, veog = raw.get_data(picks=["HEOG", "VEOG"])[:, 0] * 1e6
    assert heog == pytest.approx(4 - 3) and veog == pytest.approx(5 - 1)


def _scores(n, d8=-0.03, d2=-0.06, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model, norm in (("lr", "N2"), ("lda", "N1")):
        for i in range(n):
            base = 0.62 + rng.normal(0, 0.04)
            for b, shift in (("B1", 0.0), ("B8", d8), ("B2", d2)):
                rows.append({"subject": f"sub-{i + 1:03d}", "model": model,
                             "config": f"{b}-W1-{norm}-C1", "n_folds": 50,
                             "bacc": base + shift + rng.normal(0, 0.02)})  # fmt: skip
    return pd.DataFrame(rows)


def test_replication_tests_detect_the_registered_effects():
    rep = load_script("pursue_replication")
    scores = _scores(60)
    ids = sorted(scores.subject.unique())
    out = rep.replication_tests(scores, "lr", ids).set_index("test")
    assert out.loc["R1", "mean"] == pytest.approx(-0.03, abs=0.01)
    assert out.loc["R2", "mean"] == pytest.approx(0.06, abs=0.01)
    assert out.loc[["R1", "R2"], "replicated"].all()
    assert (out.loc[["R1", "R2"], "p_holm"] >= out.loc[["R1", "R2"], "p"]).all()
    assert out.loc["baseline > 0.5", "p"] < 1e-6
    rep.check_complete(scores, ids)


def test_replication_fails_when_an_effect_has_the_wrong_sign():
    rep = load_script("pursue_replication")
    scores = _scores(60, d8=+0.03)
    out = rep.replication_tests(scores, "lr", sorted(scores.subject.unique())).set_index("test")
    assert not out.loc["R1", "replicated"] and out.loc["R2", "replicated"]
    with pytest.raises(SystemExit, match="incomplete"):
        rep.check_complete(scores[scores.config != "B2-W1-N2-C1"], ["sub-001"])


def test_subject_table_applies_the_trial_minimum(tmp_path):
    import json

    rep = load_script("pursue_replication")
    for i, (code, rel, unrel) in enumerate([("1001", 60, 60), ("2010", 29, 60), ("3001", 30, 31)]):
        folder = tmp_path / f"sub-{i + 1:03d}"
        folder.mkdir()
        meta = {"subject": folder.name, "subject_code": code, "n_related": rel,
                "n_unrelated": unrel}  # fmt: skip
        (folder / "preproc.json").write_text(json.dumps(meta))
    table = rep.subject_table(tmp_path)
    assert table.included.tolist() == [True, False, True]
    assert table.site.tolist() == [pursue.SITES[k] for k in "123"]


class Files(BaseHTTPRequestHandler):
    """Serves ``content``; the first request for ``flaky`` returns corrupted bytes."""

    content: dict = {}
    calls: dict = {}

    def do_GET(self):
        name = self.path.lstrip("/")
        Files.calls[name] = Files.calls.get(name, 0) + 1
        body = Files.content[name]
        if name == "flaky" and Files.calls[name] == 1:
            body = b"x" * len(body)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_manifest_download_verifies_retries_and_skips(tmp_path, monkeypatch):
    fm = load_script("fetch_manifest")
    monkeypatch.setattr(fm, "BACKOFF_START", 0)
    Files.content = {"a.txt": b"hello\n", "sub/data.set": b"\x00\x01" * 1000, "flaky": b"ok" * 10}
    Files.calls = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Files)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}"
    git = "ce013625030ba8dba906f756967f9e9ca394464a"  # `git hash-object` of b"hello\n"
    manifest = [
        {"path": "a.txt", "size": 6, "checksum_algorithm": "git", "checksum": git,
         "url": f"{url}/a.txt"},
        {"path": "sub/data.set", "size": 2000, "checksum_algorithm": "sha256",
         "checksum": hashlib.sha256(Files.content["sub/data.set"]).hexdigest(),
         "url": f"{url}/sub/data.set"},
        {"path": "flaky", "size": 20, "checksum_algorithm": "sha256",
         "checksum": hashlib.sha256(b"ok" * 10).hexdigest(), "url": f"{url}/flaky"},
    ]  # fmt: skip
    path = tmp_path / "manifest.json"
    path.write_text(__import__("json").dumps(manifest))
    out = tmp_path / "data"
    assert fm.main([str(path), str(out), "--workers", "2"]) == 0
    assert (out / "sub" / "data.set").read_bytes() == Files.content["sub/data.set"]
    assert Files.calls["flaky"] == 2 and not list(out.rglob("*.part"))
    assert fm.main([str(path), str(out)]) == 0  # second run: everything verified, no download
    assert Files.calls == {"a.txt": 1, "sub/data.set": 1, "flaky": 2}
    bad = [dict(manifest[0], checksum="0" * 40)]
    path.write_text(__import__("json").dumps(bad))
    monkeypatch.setattr(fm, "ATTEMPTS", 2)
    assert fm.main([str(path), str(tmp_path / "other")]) == 1
    httpd.shutdown()


def test_replication_stage_is_the_three_registered_configurations():
    from n400fp.study import stage_configs

    codes = [c.code for c in stage_configs("replication")]
    assert codes == ["B1-W1-N2-C1", "B8-W1-N2-C1", "B2-W1-N2-C1"]
