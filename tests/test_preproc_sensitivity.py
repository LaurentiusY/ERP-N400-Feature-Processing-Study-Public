"""Preprocessing sensitivity analysis (guide 13.11): comparison tables and completeness checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n400fp.joint import TASKS
from n400fp.study import full_factorial, model_configs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preproc_sensitivity.py"


def load_script():
    spec = importlib.util.spec_from_file_location("preproc_sensitivity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(shift: float, n: int, seed: int) -> pd.DataFrame:
    """LR on all 640 configurations, LDA on its 480; waveforms (B1, B8) 10 points better."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in ("lr", "lda"):
        for cfg in model_configs(full_factorial(), model):
            base = 0.5 + (0.1 if cfg.B in ("B1", "B8") else 0.0) + shift
            for i in range(n):
                rows.append({"subject": f"sub-{i + 1:03d}", "model": model, "config": cfg.code,
                             **cfg._asdict(), "bacc": base + rng.normal(0, 0.01)})  # fmt: skip
    return pd.DataFrame(rows)


def test_comparison_tables_for_both_models():
    ps = load_script()
    main = {t: _scores(0.0, 8, k) for k, t in enumerate(TASKS)}
    alt = ps.with_prespecified(
        main, {"scene": _scores(-0.01, 8, 10), "sentence": _scores(0.005, 8, 11)}
    )
    assert alt["erpcore"] is main["erpcore"]  # ERP CORE enters both joint tests unchanged
    with pytest.raises(ValueError, match="reruns"):
        ps.with_prespecified(main, {"scene": main["scene"]})
    for t in ("scene", "sentence"):
        ps.check_complete(main[t], alt[t], t)
    for model in ps.MODELS:
        joint = ps.joint_table(main, alt, model)
        assert len(joint) == 11 and joint.category_changed.dtype == bool
        p3 = joint.set_index("contrast").loc["P3"]
        assert p3.estimate_main == pytest.approx(0.1, abs=0.01)
        assert p3.estimate_prespecified == pytest.approx(0.1, abs=0.01)
        assert not p3.category_changed
    planned = ps.planned_table(main["scene"], alt["scene"], "lda")
    assert len(planned) == 11 and set(planned.model) == {"lda"}
    assert {"mean_main", "mean_prespecified", "significant_changed", "sign_changed"} <= set(
        planned.columns
    )
    for model, base in (("lr", "B1-W1-N2-C1"), ("lda", "B1-W1-N1-C1")):
        row = ps.task_summary("scene", main["scene"], alt["scene"], model)
        assert row["baseline_config"] == base
        assert row["baseline_diff"] == pytest.approx(-0.01, abs=0.01)
        assert row["n_subjects"] == 8


def test_main_reads_both_runs_and_writes_only_its_folder(monkeypatch, tmp_path, capsys):
    """Main scores from ``<results>/<task>``, reruns from ``preproc_prespecified/<task>``,
    whatever ``N400_DATASET`` and ``N400_PREPROC`` the shell still has."""
    ps = load_script()
    roots = (tmp_path, tmp_path / "scene", tmp_path / "sentence")  # erpcore, scene, sentence
    for k, root in enumerate(roots):
        (root / "reports" / "full").mkdir(parents=True, exist_ok=True)
        _scores(0.0, 4, k).to_csv(root / "reports" / "full" / "subject_scores.csv", index=False)
    for k, task in enumerate(("scene", "sentence")):
        rerun = tmp_path / "preproc_prespecified" / task / "reports" / "full"
        rerun.mkdir(parents=True)
        _scores(-0.01, 4, k + 10).to_csv(rerun / "subject_scores.csv", index=False)
    before = {p for p in tmp_path.rglob("*")}
    monkeypatch.setenv("N400_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("N400_DATASET", "sentence")
    monkeypatch.setenv("N400_PREPROC", "prespecified")
    ps.main()
    out = tmp_path / "preproc_sensitivity"
    created = {p for p in tmp_path.rglob("*")} - before
    assert created == {out} | set(out.iterdir())
    assert {p.name for p in out.iterdir()} == {
        "summary.csv", "planned_scene.csv", "planned_sentence.csv", "joint.csv"
    }  # fmt: skip
    summary = pd.read_csv(out / "summary.csv")
    assert summary.baseline_diff.to_numpy() == pytest.approx(-0.01, abs=0.02)
    assert len(pd.read_csv(out / "joint.csv")) == 22  # 11 contrasts x 2 models
    assert "RESULT:" in capsys.readouterr().out


def test_incomplete_rerun_is_rejected():
    ps = load_script()
    main = _scores(0.0, 3, 0)
    one_lda = (main.subject == "sub-002") & (main.model == "lda") & (main.config == "B1-W1-N1-C1")
    with pytest.raises(SystemExit, match="lda: incomplete"):
        ps.check_complete(main, main[~one_lda], "scene")
    with pytest.raises(SystemExit, match="missing subjects"):
        ps.check_complete(main, main[main.subject != "sub-003"], "scene")
    with pytest.raises(SystemExit, match=r"only in the pre-specified run \['sub-003'\]"):
        ps.check_complete(main[main.subject != "sub-003"], main, "scene")
    with pytest.raises(SystemExit, match="missing model"):
        ps.check_complete(main, main[main.model == "lr"], "sentence")
    with pytest.raises(SystemExit, match="incomplete main"):
        ps.check_complete(main[~one_lda], main, "scene")
