import numpy as np

from n400fp.config import DecodeConfig, PermutationConfig
from n400fp.decode import outer_splits
from n400fp.features import build_features, flatten
from n400fp.permutation import permutation_test
from n400fp.study import Config
from n400fp.timecourse import timecourse

FAST = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))


def _baseline(data):
    x4, _ = build_features(data, Config("B1", "W1", "N2", "C1"))
    return flatten(x4), tuple(x4.shape[1:])


def test_permutation_small_p_with_effect(subject):
    x, shape = _baseline(subject)
    folds = outer_splits(subject.y, "sub-001")[:10]
    res = permutation_test(x, subject.y, folds, 0.1, "N2", shape, PermutationConfig(99, seed=1))
    assert res["observed"] > res["null_95"]
    assert res["p"] == 0.01  # (1 + 0) / (1 + 99)
    assert abs(res["null_mean"] - 0.5) < 0.05


def test_permutation_large_p_without_effect(null_subject):
    x, shape = _baseline(null_subject)
    folds = outer_splits(null_subject.y, "sub-002")[:10]
    res = permutation_test(
        x, null_subject.y, folds, 0.1, "N2", shape, PermutationConfig(99, seed=1)
    )
    assert res["p"] > 0.05


def test_permutation_reproducible(subject):
    x, shape = _baseline(subject)
    folds = outer_splits(subject.y, "sub-001")[:10]
    cfg = PermutationConfig(20, seed=5)
    a = permutation_test(x, subject.y, folds, 0.1, "N2", shape, cfg)
    b = permutation_test(x, subject.y, folds, 0.1, "N2", shape, cfg, n_jobs=2)
    assert a == b


def test_timecourse_bins_and_effect_timing(subject):
    splits = outer_splits(subject.y, "sub-001", FAST)
    tc = timecourse(subject, splits, FAST)
    assert set(tc.channels) == {"C1", "C5"}
    starts = sorted(tc.bin_start.unique())
    assert len(starts) == 16 and np.isclose(starts[0], 0.0) and np.isclose(starts[-1], 0.75)
    curve = tc[tc.channels == "C1"].groupby("bin_start").bacc.mean()
    assert curve.loc[0.35:0.45].mean() > curve.loc[0.0:0.1].mean() + 0.1
