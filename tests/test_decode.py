import numpy as np
import pandas as pd
import pytest

from n400fp.config import DecodeConfig
from n400fp.decode import evaluate, get_splits, load_splits, make_estimator, outer_splits
from n400fp.features import build_features, flatten
from n400fp.run_subject import modal_c, plan, run_subject, subject_means
from n400fp.study import Config, full_factorial, pilot

FAST = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))


def test_outer_splits_reproducible_and_stratified(subject):
    a = outer_splits(subject.y, "sub-001")
    b = outer_splits(subject.y, "sub-001")
    c = outer_splits(subject.y, "sub-002")
    assert len(a) == 50
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b, strict=True))
    assert not all(np.array_equal(x[1], z[1]) for x, z in zip(a, c, strict=True))
    for r in range(5):
        tests = np.concatenate([te for _, te in a[r * 10 : (r + 1) * 10]])
        assert np.array_equal(np.sort(tests), np.arange(120))  # each repeat covers all trials
    for tr, te in a:
        assert set(tr).isdisjoint(te)
        assert abs(subject.y[te].mean() - 0.5) <= 0.1


def test_splits_saved_once_and_reused(subject, tmp_path):
    path = tmp_path / "splits" / "sub-001.npz"
    first = get_splits(subject.y, "sub-001", path)
    again = get_splits(subject.y, "sub-001", path)
    loaded = load_splits(path)
    for x, y, z in zip(first, again, loaded, strict=True):
        assert np.array_equal(x[1], y[1]) and np.array_equal(x[1], z[1])


def test_baseline_above_chance_and_shuffled_near_chance(subject):
    x4, _ = build_features(subject, Config("B1", "W1", "N2", "C1"))
    x, shape = flatten(x4), tuple(x4.shape[1:])
    splits = outer_splits(subject.y, "sub-001", FAST)
    real = np.mean([r["bacc"] for r in evaluate(x, subject.y, splits, "lr", "N2", shape, FAST)])
    assert real > 0.7
    shuffled_y = np.random.default_rng(3).permutation(subject.y)
    shuf = np.mean([r["bacc"] for r in evaluate(x, shuffled_y, splits, "lr", "N2", shape, FAST)])
    assert abs(shuf - 0.5) < 0.15


def test_null_subject_near_chance(null_subject):
    x4, _ = build_features(null_subject, Config("B1", "W1", "N2", "C1"))
    splits = outer_splits(null_subject.y, "sub-002", FAST)
    rows = evaluate(flatten(x4), null_subject.y, splits, "lda", "N1", tuple(x4.shape[1:]), FAST)
    assert abs(np.mean([r["bacc"] for r in rows]) - 0.5) < 0.15


def test_selected_c_recorded_and_from_grid(subject):
    x4, _ = build_features(subject, Config("B1", "W2", "N2", "C2"))
    splits = outer_splits(subject.y, "sub-001", FAST)
    rows = evaluate(flatten(x4), subject.y, splits, "lr", "N2", tuple(x4.shape[1:]), FAST)
    assert all(r["C"] in FAST.c_grid for r in rows)
    lda = evaluate(flatten(x4), subject.y, splits, "lda", "N1", tuple(x4.shape[1:]), FAST)
    assert all(np.isnan(r["C"]) for r in lda)


def test_inner_tuning_ignores_test_trials(subject):
    """C is chosen on the training fold only: altering test trials cannot change it."""
    x4, _ = build_features(subject, Config("B1", "W1", "N2", "C1"))
    x, shape = flatten(x4), tuple(x4.shape[1:])
    tr, te = outer_splits(subject.y, "sub-001", FAST)[0]
    x2 = x.copy()
    x2[te] = np.random.default_rng(0).standard_normal(x2[te].shape) * 100
    a = make_estimator("lr", "N4", shape, FAST).fit(x[tr], subject.y[tr])
    b = make_estimator("lr", "N4", shape, FAST).fit(x2[tr], subject.y[tr])
    assert a.best_params_ == b.best_params_
    assert np.allclose(a.decision_function(x[te]), b.decision_function(x[te]))


def test_plan_groups_by_features():
    groups = plan(full_factorial(), ("lr", "lda"))
    assert len(groups) == 8 * 4 * 5
    assert sum(len(v) for v in groups.values()) == 640 + 480
    assert all(len(v) == 4 + 3 for v in groups.values())


def test_run_subject_writes_and_resumes(subject, tmp_path):
    configs = pilot()[:3]
    df = run_subject(
        "sub-001", "pilot", configs=configs, results_root=tmp_path, dcfg=FAST, data=subject
    )
    files = sorted((tmp_path / "pilot" / "sub-001").glob("*.parquet"))
    assert files
    n_lr = len(configs)
    n_lda = len({c._replace(N="N1" if c.N == "N2" else c.N) for c in configs})
    assert len(df) == (n_lr + n_lda) * 5
    mtimes = [f.stat().st_mtime_ns for f in files]
    df2 = run_subject(
        "sub-001", "pilot", configs=configs, results_root=tmp_path, dcfg=FAST, data=subject
    )
    assert [f.stat().st_mtime_ns for f in files] == mtimes  # nothing recomputed
    pd.testing.assert_frame_equal(
        df.sort_values(["model", "config", "fold"]).reset_index(drop=True),
        df2.sort_values(["model", "config", "fold"]).reset_index(drop=True),
    )
    means = subject_means(df)
    assert (means.n_folds == 5).all()
    assert modal_c(df, configs[0].code) in FAST.c_grid


@pytest.mark.parametrize("model", ["lr", "lda"])
def test_estimators_build(model):
    est = make_estimator(model, "N2", (1, 3, 4), FAST)
    assert hasattr(est, "fit")
