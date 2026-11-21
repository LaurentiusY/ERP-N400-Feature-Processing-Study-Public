import numpy as np
import pandas as pd
import pytest
from scipy import stats

from n400fp.joint import (
    CATEGORIES,
    TASKS,
    classify,
    config_agreement,
    effect_size_table,
    joint_contrasts,
    joint_test,
)
from n400fp.stats import PLANNED, levels_for
from n400fp.study import full_factorial, model_configs


def _series(values, prefix="sub-"):
    return pd.Series(values, index=[f"{prefix}{i:03d}" for i in range(1, len(values) + 1)])


def test_joint_estimate_and_se_by_hand():
    rng = np.random.default_rng(0)
    de = _series(rng.normal(0.02, 0.03, 40), "e-")
    ds = _series(rng.normal(0.01, 0.02, 40))
    dt = _series(rng.normal(0.03, 0.02, 40))
    r = joint_test(de, ds, dt)
    a = (ds + dt) / 2
    est = (de.mean() + ds.mean() + dt.mean()) / 3  # equal weights
    v_e, v_f = de.var() / (9 * 40), 4 * a.var() / (9 * 40)
    df = (v_e + v_f) ** 2 / (v_e**2 / 39 + v_f**2 / 39)
    assert r["estimate"] == pytest.approx(est)
    assert r["se"] == pytest.approx(np.sqrt(v_e + v_f))
    assert r["df"] == pytest.approx(df)
    assert r["p"] == pytest.approx(2 * stats.t.sf(abs(est / np.sqrt(v_e + v_f)), df))
    assert r["ci_low"] < est < r["ci_high"]


def test_shared_participants_are_not_treated_as_independent():
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.03, 40)  # same people: strongly correlated task differences
    de = _series(rng.normal(0.0, 0.03, 40), "e-")
    ds = _series(0.01 + common + rng.normal(0, 0.003, 40))
    dt = _series(0.01 + common + rng.normal(0, 0.003, 40))
    r = joint_test(de, ds, dt)
    naive = np.sqrt((de.var() + ds.var() + dt.var()) / (9 * 40))  # wrongly independent
    assert r["se"] > naive * 1.15  # positive correlation must widen the interval


def test_heterogeneity():
    rng = np.random.default_rng(2)
    same = joint_test(*(_series(rng.normal(0.02, 0.02, 40), p) for p in ("e-", "s-", "s-")))
    assert same["heterogeneity_p"] > 0.01
    de = _series(rng.normal(0.05, 0.02, 40), "e-")
    ds = _series(rng.normal(-0.03, 0.02, 40))
    dt = _series(rng.normal(-0.03, 0.02, 40))
    assert joint_test(de, ds, dt)["heterogeneity_p"] < 1e-6
    ds2 = _series(rng.normal(0.05, 0.02, 40))
    dt2 = ds2 - 0.08 + rng.normal(0, 0.005, 40)  # scene vs sentence differ, paired
    assert joint_test(de, ds2, dt2)["heterogeneity_p"] < 1e-6


@pytest.mark.parametrize(
    "p, signs, het, expected",
    [
        (0.001, [0.1, 0.2, 0.05], 0.5, CATEGORIES[0]),
        (0.001, [-0.1, -0.2, -0.05], 0.01, CATEGORIES[1]),
        (0.2, [0.1, -0.2, 0.05], 0.001, CATEGORIES[2]),
        (0.001, [0.1, -0.2, 0.05], 0.001, CATEGORIES[2]),
        (0.2, [0.1, 0.2, 0.05], 0.5, CATEGORIES[3]),
        (0.001, [0.1, -0.2, 0.05], 0.5, CATEGORIES[3]),
    ],
)
def test_classify(p, signs, het, expected):
    assert classify(p, signs, het) == expected


def _scores(effect_b1: float, n: int, seed: int) -> pd.DataFrame:
    """Complete synthetic subject scores (both models) with a waveform advantage."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in ("lr", "lda"):
        for cfg in model_configs(full_factorial(), model):
            base = 0.5 + (effect_b1 if cfg.B in ("B1", "B8") else 0.0)
            for i in range(n):
                rows.append({"subject": f"sub-{i + 1:03d}", "model": model, "config": cfg.code,
                             **cfg._asdict(), "bacc": base + rng.normal(0, 0.01)})  # fmt: skip
    return pd.DataFrame(rows)


def test_joint_contrasts_end_to_end():
    scores = {t: _scores(0.1, 12, k) for k, t in enumerate(TASKS)}
    out = joint_contrasts(scores, "lr")
    assert out.contrast.tolist() == [code for code, _, _ in PLANNED]
    assert set(out.category) <= set(CATEGORIES)
    p3 = out.set_index("contrast").loc["P3"]  # B1 - B2: +0.1 in every task
    assert p3.estimate == pytest.approx(0.1, abs=0.01)
    assert p3.category == "general"
    assert (out.p_holm >= out.p - 1e-12).all()
    for task in TASKS:
        assert p3[f"{task}_mean"] == pytest.approx(0.1, abs=0.01)
    assert len(joint_contrasts(scores, "lda")) == 11  # LDA levels (N1 for N2) handled
    assert levels_for("lda")["N"] == ("N1", "N3", "N4")


def test_effect_sizes_and_config_agreement():
    anova = {
        t: pd.DataFrame({"effect": ["B", "W"], "ges": [0.3 + k / 10, 0.02]})
        for k, t in enumerate(TASKS)
    }
    table = effect_size_table(anova).set_index("effect")
    assert table.loc["B", "mean"] == pytest.approx(0.4)
    tests = {
        t: pd.DataFrame({"model": "lr", "config": ["a", "b", "c"], "mean_bacc": [0.7, 0.6, 0.5]})
        for t in TASKS
    }
    agreement = config_agreement(tests, "lr")
    assert np.allclose(agreement["correlations"].spearman, 1.0)
    assert agreement["ranking"].config.tolist() == ["a", "b", "c"]
