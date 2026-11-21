import itertools

import numpy as np
import pandas as pd
import pytest

from n400fp.stats import (
    _marginal,
    anova_table,
    art_anova,
    config_tests,
    export_marginal_tables,
    fdr_bh,
    holm,
    levels_for,
    loso_best_band,
    pilot_report,
    planned_contrasts,
    rm_anova,
    summarize,
    timecourse_tests,
    to_array,
)
from n400fp.study import FACTORS, full_factorial, model_configs


def test_holm_and_bh_by_hand():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(fdr_bh(p), [0.02, 0.04, 0.04, 0.02])


def test_summarize_ci_and_dz():
    from scipy import stats

    d = np.array([0.02, 0.05, 0.01, 0.04, 0.03, 0.06, 0.00, 0.05])
    s = summarize(d, alternative="greater")
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = stats.t.ppf(0.975, len(d) - 1) * se
    assert np.isclose(s["ci_low"], d.mean() - half) and np.isclose(s["ci_high"], d.mean() + half)
    assert np.isclose(s["dz"], d.mean() / d.std(ddof=1))
    assert np.isclose(s["p"], stats.ttest_1samp(d, 0, alternative="greater").pvalue)


def _one_way_by_hand(y):
    n, k = y.shape
    grand = y.mean()
    ss_a = n * ((y.mean(0) - grand) ** 2).sum()
    ss_s = k * ((y.mean(1) - grand) ** 2).sum()
    ss_tot = ((y - grand) ** 2).sum()
    ss_err = ss_tot - ss_a - ss_s
    f = (ss_a / (k - 1)) / (ss_err / ((k - 1) * (n - 1)))
    return f, ss_a, ss_err, ss_s


def test_rm_anova_one_way_matches_textbook(rng):
    y = rng.normal(size=(12, 4)) + np.array([0.0, 0.3, 0.6, 0.2])
    out = rm_anova(y, ("A",))
    f, ss_a, ss_err, ss_s = _one_way_by_hand(y)
    assert np.isclose(out.F[0], f)
    assert out.df1[0] == 3 and out.df2[0] == 33
    assert np.isclose(out.ges[0], ss_a / (ss_a + ss_s + ss_err))
    assert 1 / 3 <= out.gg_epsilon[0] <= 1


def test_rm_anova_two_levels_has_epsilon_one_and_equals_paired_t(rng):
    from scipy import stats

    y = rng.normal(size=(15, 2)) + [0, 0.4]
    out = rm_anova(y, ("A",))
    t = stats.ttest_rel(y[:, 1], y[:, 0])
    assert out.gg_epsilon[0] == 1.0
    assert np.isclose(out.F[0], t.statistic**2) and np.isclose(out.p_gg[0], t.pvalue)


def test_rm_anova_two_way_matches_statsmodels(rng):
    sm = pytest.importorskip("statsmodels.stats.anova")
    y = rng.normal(size=(10, 3, 4))
    y[:, 1] += 0.5
    y[:, :, 2] += 0.3
    out = rm_anova(y, ("A", "B")).set_index("effect")
    rows = [
        {"s": s, "A": a, "B": b, "v": y[s, a, b]}
        for s, a, b in itertools.product(range(10), range(3), range(4))
    ]
    ref = sm.AnovaRM(pd.DataFrame(rows), "v", "s", within=["A", "B"]).fit().anova_table
    for eff, name in (("A", "A"), ("B", "B"), ("A:B", "A:B")):
        assert np.isclose(out.loc[eff, "F"], ref.loc[name, "F Value"])
        assert out.loc[eff, "df2"] == ref.loc[name, "Den DF"]


def test_anova_table_confirmatory_effects(rng):
    y = rng.normal(size=(8, 2, 2, 3, 2))
    out = anova_table(y)
    assert len(out) == 15
    assert out.confirmatory.sum() == 10
    assert out.loc[~out.confirmatory, "p_bh"].isna().all()


def test_art_detects_main_effect_and_respects_null(rng):
    y = rng.normal(size=(20, 3, 2)) * 0.2
    y[:, 2, :] += 1.0  # strong effect of A only
    out = art_anova(y, ("A", "B")).set_index("effect")
    assert out.loc["A", "p_gg"] < 0.001
    assert out.loc["B", "p_gg"] > 0.01


def _fake_scores(rng, model="lr", n_sub=12, boost=None):
    levels = levels_for(model)
    rows = []
    configs = model_configs(full_factorial(), model)
    for s in range(n_sub):
        for c in configs:
            v = 0.55 + rng.normal(0, 0.02)
            for key, delta in (boost or {}).items():
                if getattr(c, key[0]) == key[1]:
                    v += delta
            rows.append({"subject": f"sub-{s + 1:03d}", "model": model, "config": c.code,
                         "B": c.B, "W": c.W, "N": c.N, "C": c.C, "bacc": v})  # fmt: skip
    assert len(configs) == np.prod([len(v) for v in levels.values()])
    return pd.DataFrame(rows)


def test_to_array_rejects_incomplete_design(rng):
    scores = _fake_scores(rng, n_sub=3)
    _, y = to_array(scores, levels_for("lr"))
    assert y.shape == (3, 8, 4, 4, 5)
    with pytest.raises(ValueError):
        to_array(scores.iloc[1:], levels_for("lr"))


def test_planned_contrasts_by_hand(rng):
    scores = _fake_scores(rng, boost={("B", "B8"): 0.03, ("C", "C5"): -0.01})
    out = planned_contrasts(scores, "lr").set_index("contrast")
    assert list(out.index) == [f"P{i}" for i in range(1, 12)]
    _, y = to_array(scores, levels_for("lr"))
    lv = levels_for("lr")
    p1 = _marginal(y, lv, B="B8") - _marginal(y, lv, B="B1")
    assert np.isclose(out.loc["P1", "mean"], p1.mean())
    assert out.loc["P1", "p_holm"] < 0.05
    p11 = (_marginal(y, lv, C="C1", W="W3") - _marginal(y, lv, C="C5", W="W3")) - (
        _marginal(y, lv, C="C1", W="W2") - _marginal(y, lv, C="C5", W="W2")
    )
    assert np.isclose(out.loc["P11", "mean"], p11.mean())
    assert np.allclose(out.p_holm, np.minimum(1, out.p_holm))


def test_planned_contrasts_lda_uses_n1(rng):
    scores = _fake_scores(rng, model="lda", n_sub=6)
    out = planned_contrasts(scores, "lda").set_index("contrast")
    assert "N1" in out.loc["P6", "note"]


def test_loso_best_band_excludes_own_subject():
    bands = np.array([[0.9, 0.0], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]])
    pick = loso_best_band(bands)
    # subject 0 alone favours column 0; without it, column 1 wins
    assert pick[0] == 1
    assert list(pick[1:]) == [0, 0, 0]


def test_config_tests_and_export(rng, tmp_path):
    scores = _fake_scores(rng, n_sub=5)
    out = config_tests(scores)
    assert len(out) == 640 and (out.p_holm >= out.p).all()
    files = export_marginal_tables(scores, "lr", tmp_path)
    assert len(files) == 4 + 6
    b = pd.read_csv(tmp_path / "lr_B.csv")
    assert len(b) == 5 * 8 and set(b.columns) == {"subject", "B", "acc"}
    bw = pd.read_csv(tmp_path / "lr_B_W.csv")
    assert len(bw) == 5 * 8 * 4


def test_pilot_report_flags_c_at_grid_edge():
    folds = pd.DataFrame(
        {
            "subject": ["a"] * 4,
            "model": ["lr"] * 4,
            "config": ["B1-W1-N2-C1"] * 4,
            "bacc": [0.6] * 4,
            "seconds": [1.0] * 4,
            "C_selected": [100.0, 100.0, 1.0, 1.0],
        }
    )
    out = pilot_report(folds, (1e-3, 1.0, 100.0))
    assert out.C_edge_warning[0] and np.isclose(out.frac_C_max[0], 0.5)


def test_timecourse_tfce_reproducible_and_null(rng):
    pytest.importorskip("mne")
    rows = []
    for s in range(20):
        for k in range(16):
            signal = 0.08 if 6 <= k <= 9 else 0.0
            rows.append({"subject": s, "channels": "C1", "bin_start": k * 0.05,
                         "bacc": 0.5 + signal + rng.normal(0, 0.03)})  # fmt: skip
            rows.append({"subject": s, "channels": "C5", "bin_start": k * 0.05,
                         "bacc": 0.5 + rng.normal(0, 0.03)})  # fmt: skip
    tc = pd.DataFrame(rows)
    a = timecourse_tests(tc, seed=1, n_permutations=256)
    b = timecourse_tests(tc, seed=1, n_permutations=256)
    pd.testing.assert_frame_equal(a, b)
    c1 = a[a.channels == "C1"].set_index("bin_start")
    assert (c1.p_tfce.iloc[6:10] < 0.05).all()
    c5 = a[a.channels == "C5"]
    assert (c5.p_tfce > 0.05).mean() > 0.8


def test_factors_unchanged():
    assert {k: len(v) for k, v in FACTORS.items()} == {"B": 8, "W": 4, "N": 4, "C": 5}
