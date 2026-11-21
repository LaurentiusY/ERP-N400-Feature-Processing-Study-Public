"""C-grid sensitivity analysis: separate results folder, main folds reused, comparison tables."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n400fp.config import C_GRIDS, DecodeConfig
from n400fp.joint import TASKS
from n400fp.run_subject import run_subject, splits_path
from n400fp.study import full_factorial, pilot

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "c_grid_sensitivity.py"
FAST = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))


def load_script():
    spec = importlib.util.spec_from_file_location("c_grid_sensitivity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extended_grid_contains_the_study_grid_and_extends_both_ends():
    study, extended = C_GRIDS["study"], C_GRIDS["extended"]
    assert DecodeConfig().c_grid == study
    assert set(study) < set(extended)
    assert min(extended) < min(study) and max(extended) > max(study)
    assert list(extended) == sorted(extended)


def test_sensitivity_run_reuses_main_folds_and_writes_only_its_own_folder(subject, tmp_path):
    main_root, ext_root = tmp_path / "main", tmp_path / "main" / "c_grid_extended"
    configs = pilot()[:2]
    run_subject("sub-001", "pilot", ("lr",), configs, results_root=main_root, dcfg=FAST,
                data=subject)  # fmt: skip
    main_files = sorted(p for p in main_root.rglob("*") if ext_root not in p.parents)
    mtimes = [p.stat().st_mtime_ns for p in main_files]
    wide = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(1e-4, 0.01, 1.0))
    df = run_subject("sub-001", "pilot", ("lr",), configs, results_root=ext_root, dcfg=wide,
                     data=subject, splits_root=main_root)  # fmt: skip
    assert not splits_path("sub-001", ext_root).exists()  # folds read from the main run
    assert (ext_root / "pilot" / "sub-001").is_dir()
    assert [p.stat().st_mtime_ns for p in main_files] == mtimes  # main results untouched
    assert set(df.C_selected) <= {1e-4, 0.01, 1.0}
    assert len(df.groupby(["config", "repeat", "fold"])) == len(configs) * 5


def test_sensitivity_run_needs_the_main_folds(subject, tmp_path):
    with pytest.raises(FileNotFoundError, match="no saved folds"):
        run_subject("sub-001", "pilot", ("lr",), pilot()[:1], results_root=tmp_path / "ext",
                    dcfg=FAST, data=subject, splits_root=tmp_path / "missing")  # fmt: skip


def _scores(shift: float, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in full_factorial():
        base = 0.5 + (0.1 if cfg.B in ("B1", "B8") else 0.0) + shift
        for i in range(n):
            bacc = base + rng.normal(0, 0.01)
            rows.append({"subject": f"sub-{i + 1:03d}", "model": "lr", "config": cfg.code,
                         **cfg._asdict(), "bacc": bacc, "n_folds": 50})  # fmt: skip
    return pd.DataFrame(rows)


def test_comparison_tables():
    cs = load_script()
    main = {t: _scores(0.0, 8, k) for k, t in enumerate(TASKS)}
    ext = {t: _scores(-0.002, 8, k + 10) for k, t in enumerate(TASKS)}
    for t in TASKS:
        cs.check_complete(main[t], ext[t], t)
    joint = cs.joint_table(main, ext)
    assert len(joint) == 11 and joint.category_changed.dtype == bool
    p3 = joint.set_index("contrast").loc["P3"]
    assert p3.estimate_study == pytest.approx(0.1, abs=0.01)
    assert p3.estimate_extended == pytest.approx(0.1, abs=0.01)
    planned = cs.planned_table(main["erpcore"], ext["erpcore"])
    assert {"mean_study", "mean_extended", "significant_changed"} <= set(planned.columns)
    folds = pd.DataFrame({"C_selected": [1e-5, 1e-3, 1.0, 1e4, 1e-4]})
    edges = cs.edge_fractions(folds)
    assert edges["frac_below_study_grid"] == pytest.approx(0.4)
    assert edges["frac_at_extended_min"] == pytest.approx(0.2)
    assert edges["frac_above_study_grid"] == pytest.approx(0.2)
    row = cs.task_summary("erpcore", main["erpcore"], ext["erpcore"], folds)
    assert row["baseline_diff"] == pytest.approx(-0.002, abs=0.01)


def test_incomplete_extended_run_is_rejected():
    cs = load_script()
    main = _scores(0.0, 3, 0)
    ext = main[~((main.subject == "sub-002") & (main.config == "B1-W1-N2-C1"))]
    with pytest.raises(SystemExit, match="incomplete"):
        cs.check_complete(main, ext, "scene")
    with pytest.raises(SystemExit, match="missing subjects"):
        cs.check_complete(main, main[main.subject != "sub-003"], "scene")
