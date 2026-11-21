"""Time-frequency sensitivity analysis (guide 13.10): wavelets, features, stacking, script."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from n400fp.config import TFR_CYCLES, TFR_FREQS, DecodeConfig, FeatureConfig
from n400fp.decode import evaluate, make_stacking
from n400fp.features import build_features, component_series, flatten
from n400fp.run_subject import STACKING_COLUMNS, run_subject, splits_path
from n400fp.study import (
    STACKED,
    TFR_LEVELS,
    TFR_REPRESENTATIONS,
    Config,
    components,
    full_factorial,
    stage_configs,
)
from n400fp.synthetic import make_subject
from n400fp.tfr import edge_windows, fill_epochs, morlet_segment, segments

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tfr_sensitivity.py"
FAST = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(0.01, 1.0))


@pytest.fixture(scope="module")
def tfr_subject():
    return make_subject(n_trials=80, effect_uv=4.0, seed=3, tfr=True)


@pytest.fixture(scope="module")
def null_tfr_subject():
    return make_subject(n_trials=80, effect_uv=0.0, theta_gain=0.0, seed=4, tfr=True)


def load_script():
    spec = importlib.util.spec_from_file_location("tfr_sensitivity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- design --------------------------------------------------------------------------------------


def test_frequencies_cycles_and_levels():
    assert len(TFR_FREQS) == len(TFR_CYCLES) == 12
    assert TFR_FREQS[0] == pytest.approx(2.0) and TFR_FREQS[-1] == pytest.approx(36.0)
    assert TFR_CYCLES[0] == pytest.approx(3.0) and TFR_CYCLES[-1] == pytest.approx(10.0)
    assert np.allclose(np.diff(np.log(TFR_FREQS)), np.log(18) / 11)
    assert len(TFR_REPRESENTATIONS["T1"]) == 12
    assert len(TFR_REPRESENTATIONS["T2"]) == 6  # 2.00-7.44 Hz
    assert len(TFR_REPRESENTATIONS["T3"]) == 12  # cos and sin of the same six
    assert TFR_REPRESENTATIONS["T4"] == ("wave", *TFR_REPRESENTATIONS["T1"])
    assert components("T5") == components("T4")
    assert STACKED["T5"] == ("B1", "T1")


def test_tfr_stage_is_outside_the_factorial_design():
    assert len(full_factorial()) == 640
    assert all(c.B not in TFR_LEVELS for c in full_factorial())
    codes = [c.code for c in stage_configs("tfr")]
    assert codes == [f"{b}-W1-N2-C1" for b in ("T1", "T2", "T3", "T4", "T5")]
    assert Config.parse("T5-W1-N2-C1").B == "T5"
    with pytest.raises(ValueError):
        Config.parse("T6-W1-N2-C1")


# --- wavelets on continuous data -----------------------------------------------------------------


def _raw_with(annotations):
    import mne

    info = mne.create_info(["Cz", "Pz"], 256.0, "eeg")
    raw = mne.io.RawArray(np.zeros((2, 256 * 20)), info, verbose="error")
    raw.set_annotations(mne.Annotations(*zip(*annotations, strict=True)) if annotations else None)
    return raw


def test_segments_split_at_skip_annotations():
    assert segments(_raw_with([])) == [(0, 5120)]
    raw = _raw_with([(5.0, 0.0, "boundary"), (12.0, 2.0, "BAD_ACQ_SKIP"), (3.0, 0.0, "211")])
    assert segments(raw) == [(0, 1280), (1280, 3072), (3584, 5120)]


def test_wavelet_peak_is_aligned_in_time_and_frequency():
    sfreq, n = 256.0, 256 * 20
    t = np.arange(n) / sfreq
    center = 10.0
    k = int(np.argmin(np.abs(np.asarray(TFR_FREQS) - 10.0)))
    burst = np.exp(-0.5 * ((t - center) / 0.3) ** 2) * np.cos(2 * np.pi * TFR_FREQS[k] * t)
    data = np.vstack([burst, 0.5 * burst])
    amp = np.abs(morlet_segment(data, sfreq, TFR_FREQS[k], TFR_CYCLES[k]))
    assert abs(t[np.argmax(amp[0])] - center) <= 2 / sfreq  # zero-phase: no time shift
    assert np.allclose(amp[1], 0.5 * amp[0])  # linear, channel by channel
    # the burst is strongest at its own frequency
    peaks = [np.abs(morlet_segment(data[:1], sfreq, f, c)).max() for f, c in
             zip(TFR_FREQS, TFR_CYCLES, strict=True)]  # fmt: skip
    assert int(np.argmax(peaks)) == k


def test_fill_epochs_takes_each_sample_from_its_own_segment():
    z = np.full((2, 1, 10), np.nan, complex)
    first = np.array([2, 15])
    a = np.arange(0, 20)[None, :] * (1 + 0j)  # segment 0: samples 0-19
    b = 100 + np.arange(20, 40)[None, :] * (1 + 0j)  # segment 1: samples 20-39
    assert fill_epochs(z, a, 0, first) == 0
    assert fill_epochs(z, b, 20, first) == 1  # the second window crosses into segment 1
    assert np.array_equal(z[0, 0].real, np.arange(2, 12))
    assert np.array_equal(z[1, 0].real, [15, 16, 17, 18, 19, 120, 121, 122, 123, 124])


def test_edge_windows_counts_windows_near_segment_edges():
    segs = [(0, 1000), (1000, 3000)]
    first = np.array([10, 500, 990, 1500, 2950])
    # 10: near the start; 500: clear; 990: runs past the end of segment 0 (counted once);
    # 1500: clear; 2950: ends 10 samples before the end of the recording
    assert edge_windows(first, 40, segs, 100) == 3


# --- features ------------------------------------------------------------------------------------


@pytest.mark.parametrize("b", list(TFR_REPRESENTATIONS))
def test_feature_shapes(tfr_subject, b):
    x, names = build_features(tfr_subject, Config(b, "W1", "N2", "C1"))
    assert x.shape == (80, len(TFR_REPRESENTATIONS[b]), 28, 12)
    assert len(names) == 28 and np.isfinite(x).all()


def test_amplitude_is_log_baseline_corrected_and_phase_bounded(tfr_subject):
    cfg = FeatureConfig()
    amp, times = component_series(tfr_subject, "amp:3", cfg)
    base = (times >= -0.5 - 1e-9) & (times <= -0.2 + 1e-9)
    assert np.allclose(amp[..., base].mean(axis=-1), 0.0, atol=1e-6)
    expected = np.log10(np.abs(tfr_subject.tfr[:, 3]))
    expected -= expected[..., base].mean(axis=-1, keepdims=True)
    assert np.allclose(amp, expected, atol=1e-5)
    cos, _ = component_series(tfr_subject, "cos:0", cfg)
    sin, _ = component_series(tfr_subject, "sin:0", cfg)
    assert np.allclose(cos**2 + sin**2, 1.0, atol=1e-5)
    x, _ = build_features(tfr_subject, Config("T3", "W1", "N2", "C1"))
    assert np.abs(x).max() <= 1.0 + 1e-6


def test_tfr_components_need_the_coefficients(subject):
    with pytest.raises(RuntimeError, match="not loaded"):
        build_features(subject, Config("T1", "W1", "N2", "C1"))


# --- stacking ------------------------------------------------------------------------------------


def _stack_input(data):
    blocks = [build_features(data, Config(b, "W1", "N2", "C1"))[0] for b in STACKED["T5"]]
    return np.hstack([flatten(b) for b in blocks]), [b.shape[1:] for b in blocks]


def test_stacking_parts_see_only_their_own_columns(tfr_subject):
    x, parts = _stack_input(tfr_subject)
    assert [int(np.prod(p)) for p in parts] == [336, 12 * 336]
    est = make_stacking("N2", parts, FAST).fit(x, tfr_subject.y)
    assert [m.named_steps["clf"].coef_.shape[1] for m in est.models_] == [336, 12 * 336]
    assert est.meta_.named_steps["lr"].coef_.shape == (1, 2)
    with pytest.raises(ValueError):
        est.predict(x[:, :-1])


def test_meta_model_is_fitted_on_out_of_fold_decision_values_with_the_selected_c(tfr_subject):
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    from n400fp.decode import make_estimator

    x, parts = _stack_input(tfr_subject)
    y = tfr_subject.y
    est = make_stacking("N2", parts, FAST).fit(x, y)
    cv = StratifiedKFold(FAST.inner_splits, shuffle=True, random_state=FAST.inner_seed)
    blocks = [x[:, :336], x[:, 336:]]
    oof = []
    for block, shape, c in zip(blocks, parts, est.c_, strict=True):
        tuned = make_estimator("lr", "N2", shape, FAST).fit(block, y)
        assert tuned.best_params_["clf__C"] == c
        fixed = clone(tuned.best_estimator_)
        assert fixed.named_steps["clf"].C == c  # cross-fit with the selected C, not re-tuned
        oof.append(cross_val_predict(fixed, block, y, cv=cv, method="decision_function"))
    oof = StandardScaler().fit_transform(np.column_stack(oof))
    meta = LogisticRegression(C=1.0, max_iter=FAST.lr_max_iter).fit(oof, y)
    fitted = est.meta_.named_steps["lr"]
    assert np.allclose(fitted.coef_, meta.coef_, rtol=1e-4, atol=1e-6)
    assert np.allclose(fitted.intercept_, meta.intercept_, rtol=1e-4, atol=1e-6)
    # not the in-sample decision values of the refitted block models
    in_sample = StandardScaler().fit_transform(est.decision_values(x))
    in_sample = LogisticRegression(C=1.0, max_iter=FAST.lr_max_iter).fit(in_sample, y)
    assert not np.allclose(fitted.coef_, in_sample.coef_, rtol=1e-2)


def test_stacking_keeps_a_strongly_regularized_part(tfr_subject):
    """A block tuned to a tiny C has tiny decision values; the stack must not discard it."""
    tiny = DecodeConfig(outer_splits=5, outer_repeats=1, inner_splits=3, c_grid=(1e-4,))
    x, parts = _stack_input(tfr_subject)
    est = make_stacking("N2", parts, tiny).fit(x, tfr_subject.y)
    assert np.abs(est.decision_values(x)[:, 0]).max() < 0.2  # B1 decision values: small scale
    assert est.meta_.named_steps["lr"].coef_[0, 0] > 0.5  # standardized: kept
    assert len(set(est.predict(x))) == 2


def test_stacking_matches_its_best_part_when_the_other_is_noise(tfr_subject):
    """B1 carries the effect; T1 replaced by noise must not drag the stack below B1."""
    from n400fp.decode import outer_splits

    x, parts = _stack_input(tfr_subject)
    rng = np.random.default_rng(0)
    x = np.hstack([x[:, :336], rng.standard_normal((len(x), 12 * 336))])
    y = tfr_subject.y
    splits = outer_splits(y, "sub-001", FAST)
    stack = np.mean([r["bacc"] for r in evaluate(x, y, splits, "lr", "N2", parts[0], FAST, parts)])
    alone = np.mean(
        [r["bacc"] for r in evaluate(x[:, :336], y, splits, "lr", "N2", parts[0], FAST)]
    )
    assert alone > 0.65
    assert stack > alone - 0.05


def test_stacking_is_near_chance_without_an_effect(null_tfr_subject):
    from n400fp.decode import outer_splits

    x, parts = _stack_input(null_tfr_subject)
    y = null_tfr_subject.y
    rows = evaluate(x, y, outer_splits(y, "sub-002", FAST), "lr", "N2", parts[0], FAST, parts)
    assert abs(np.mean([r["bacc"] for r in rows]) - 0.5) < 0.12
    assert all(set(STACKING_COLUMNS) <= set(r) for r in rows)
    assert all(r["C"] == r["C_part0"] for r in rows)


def test_stacking_only_for_logistic_regression(tfr_subject):
    x, parts = _stack_input(tfr_subject)
    with pytest.raises(ValueError):
        evaluate(x, tfr_subject.y, [], "lda", "N1", parts[0], FAST, parts)


def test_tfr_stage_run_reuses_main_folds_and_writes_its_own_folder(tfr_subject, tmp_path):
    main_root = tmp_path / "main"
    tfr_root = main_root / "tfr_sensitivity"
    base = [Config.parse("B1-W1-N2-C1")]
    run_subject("sub-001", "baseline", ("lr",), base, results_root=main_root, dcfg=FAST,
                data=tfr_subject)  # fmt: skip
    df = run_subject("sub-001", "tfr", ("lr",), results_root=tfr_root, dcfg=FAST,
                     data=tfr_subject, splits_root=main_root)  # fmt: skip
    assert not splits_path("sub-001", tfr_root).exists()
    assert sorted(df.B.unique()) == ["T1", "T2", "T3", "T4", "T5"]
    assert len(df) == 5 * 5
    t5 = df[df.B == "T5"]
    assert t5[STACKING_COLUMNS].notna().all().all()
    assert df[df.B != "T5"][STACKING_COLUMNS].isna().all().all()
    assert set(df.n_features[df.B == "T4"]) == {13 * 336}
    means = df.groupby("B").bacc.mean()
    assert means["T5"] > 0.6  # the waveform effect carries through the stacking


def test_missing_main_folds_are_an_error(tfr_subject, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_subject("sub-001", "tfr", ("lr",), results_root=tmp_path / "x", dcfg=FAST,
                    data=tfr_subject, splits_root=tmp_path / "none")  # fmt: skip


# --- analysis script -----------------------------------------------------------------------------


def _accuracies(rng, shift=None):
    shift = shift or {}
    subjects = {"erpcore": [f"sub-{i:03d}" for i in range(1, 41)]}
    subjects["scene"] = subjects["sentence"] = subjects["erpcore"]
    out = {}
    for task, subs in subjects.items():
        base = rng.normal(0.62, 0.04, len(subs))
        cols = {lv: base + shift.get(lv, -0.05) + rng.normal(0, 0.01, len(subs))
                for lv in ("B6", "T1", "T2", "T3", "T4", "T5")}  # fmt: skip
        out[task] = pd.DataFrame({"B1": base, **cols}, index=subs)
    return out


def test_script_tables_and_wording():
    mod = load_script()
    rng = np.random.default_rng(0)
    acc = _accuracies(rng, {"T4": -0.01, "T5": 0.0, "T1": -0.02, "B6": -0.06})
    per_task = mod.per_task_table(acc)
    assert len(per_task) == 18 and set(per_task.comparison) == {f"S{k}" for k in range(1, 7)}
    joint = mod.joint_table(acc).set_index("comparison")
    assert joint.loc["S1", "estimate"] < 0 and not joint.reliable_positive["S1"]
    assert joint.reliable_positive["S3"]  # T1 better than B6 by 0.04 in every task
    assert mod.conclusion(joint.reset_index()).startswith("Neither S1 nor S2")


def test_script_wording_when_stacking_helps():
    mod = load_script()
    acc = _accuracies(np.random.default_rng(1), {"T4": -0.01, "T5": 0.01, "T1": -0.02})
    text = mod.conclusion(mod.joint_table(acc))
    assert text.startswith("Only S2")


def test_script_wording_when_concatenation_helps_and_s3_fails():
    mod = load_script()
    acc = _accuracies(np.random.default_rng(2), {"T4": 0.02, "T5": 0.02, "T1": -0.05, "B6": -0.05})
    text = mod.conclusion(mod.joint_table(acc))
    assert text.startswith("S1 has a reliable positive effect")
    assert "S3 is not reliably positive" in text
