"""Group-level statistics (sections 6.4 and 6.5).

The analysis unit is the subject x configuration mean balanced accuracy (mean over the 50
outer folds). The design is complete and balanced, which the repeated-measures ANOVA and the
planned contrasts require; :func:`to_array` checks it.

* :func:`config_tests` - every configuration against chance (one-sided t and Wilcoxon,
  Holm within model), with 95% CI and Cohen's dz.
* :func:`planned_contrasts` - the 11 confirmatory contrasts P1-P11 (two-sided paired t on
  subject-level marginal means, Holm, Wilcoxon check, CI, dz).
* :func:`rm_anova` - four-factor within-subject repeated-measures ANOVA, computed with the
  standard formulas for a balanced design (each effect tested against its interaction with
  subjects; Greenhouse-Geisser correction; generalized eta squared), equivalent to
  ``afex::aov_ez`` univariate tests.
* :func:`art_anova` - aligned rank transform (Wobbrock et al., 2011) followed by the same
  repeated-measures ANOVA for each effect.
* :func:`timecourse_tests` - TFCE cluster permutation test of the time-resolved curves
  (as Kessler et al.'s ``cluster_test``).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .study import FACTORS, LDA_NORMALIZATIONS

FACTOR_NAMES = ("B", "W", "N", "C")


# --- multiple comparisons -----------------------------------------------------------------------


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def fdr_bh(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


# --- one-sample summaries -----------------------------------------------------------------------


def summarize(diff, alternative: str = "two-sided") -> dict:
    """Mean, 95% CI (t), Cohen's dz, t test and Wilcoxon test of ``diff`` against 0."""
    from scipy import stats

    d = np.asarray(diff, dtype=float)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    t = stats.ttest_1samp(d, 0.0, alternative=alternative)
    try:
        w = stats.wilcoxon(d, alternative=alternative).pvalue
    except ValueError:  # all differences zero
        w = np.nan
    return {
        "n": n,
        "mean": mean,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "dz": mean / sd if sd > 0 else np.nan,
        "t": float(t.statistic),
        "p": float(t.pvalue),
        "p_wilcoxon": float(w),
    }


def config_tests(scores: pd.DataFrame) -> pd.DataFrame:
    """Each model x configuration against chance (0.5), one-sided; Holm within model."""
    rows = []
    for (model, config), g in scores.groupby(["model", "config"], sort=False):
        s = summarize(g.bacc.to_numpy() - 0.5, alternative="greater")
        rows.append({"model": model, "config": config, "mean_bacc": g.bacc.mean(), **s})
    out = pd.DataFrame(rows)
    out["mean_minus_chance"] = out.pop("mean")
    out["p_holm"] = out.groupby("model").p.transform(lambda p: holm(p.to_numpy()))
    out["p_wilcoxon_holm"] = out.groupby("model").p_wilcoxon.transform(lambda p: holm(p.to_numpy()))
    return out


# --- balanced array -----------------------------------------------------------------------------


def levels_for(model: str) -> dict[str, tuple[str, ...]]:
    levels = dict(FACTORS)
    if model == "lda":
        levels["N"] = LDA_NORMALIZATIONS
    return levels


def to_array(
    scores: pd.DataFrame, levels: dict[str, tuple[str, ...]], dv: str = "bacc"
) -> tuple[list[str], np.ndarray]:
    """``(subjects, array of shape (n_subjects, L_B, L_W, L_N, L_C))``; raises if incomplete."""
    subjects = sorted(scores.subject.unique())
    shape = (len(subjects), *[len(levels[f]) for f in FACTOR_NAMES])
    arr = np.full(shape, np.nan)
    s_idx = {s: i for i, s in enumerate(subjects)}
    idx = {f: {lv: i for i, lv in enumerate(levels[f])} for f in FACTOR_NAMES}
    for row in scores.itertuples(index=False):
        key = (s_idx[row.subject], *[idx[f][getattr(row, f)] for f in FACTOR_NAMES])
        arr[key] = getattr(row, dv)
    if np.isnan(arr).any():
        missing = int(np.isnan(arr).sum())
        raise ValueError(f"incomplete design: {missing} subject x configuration cells missing")
    return subjects, arr


# --- repeated-measures ANOVA --------------------------------------------------------------------


def _orthonormal_contrasts(n_levels: int) -> np.ndarray:
    centering = np.eye(n_levels) - 1.0 / n_levels
    u, _, _ = np.linalg.svd(centering)
    return u[:, : n_levels - 1]


def effects(names=FACTOR_NAMES) -> list[tuple[str, ...]]:
    return [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)]


def _effect_scores(y: np.ndarray, effect: tuple[str, ...], names) -> np.ndarray:
    """Subject scores on the orthonormal contrasts of ``effect`` (n_subjects x df)."""
    mats = []
    for name, n_levels in zip(names, y.shape[1:], strict=True):
        if name in effect:
            mats.append(_orthonormal_contrasts(n_levels))
        else:
            mats.append(np.full((n_levels, 1), 1.0 / np.sqrt(n_levels)))
    m = mats[0]
    for nxt in mats[1:]:
        m = np.kron(m, nxt)
    return y.reshape(len(y), -1) @ m


def rm_anova(
    y: np.ndarray, names=FACTOR_NAMES, only: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Univariate repeated-measures ANOVA for a complete within-subject design.

    ``y`` has shape ``(n_subjects, L_1, ..., L_k)``. For each effect: F, df, Greenhouse-
    Geisser epsilon and corrected p, generalized eta squared. ``only`` restricts the output
    to one effect (used by the ART).
    """
    from scipy import stats

    n = len(y)
    n_cells = int(np.prod(y.shape[1:]))
    subject_means = y.reshape(n, -1).mean(axis=1)
    ss_subjects = n_cells * np.sum((subject_means - subject_means.mean()) ** 2)
    rows, ss_error_total = [], 0.0
    for eff in effects(names):
        z = _effect_scores(y, eff, names)
        k = z.shape[1]
        zbar = z.mean(axis=0)
        ss_eff = n * float(zbar @ zbar)
        ss_err = float(np.sum((z - zbar) ** 2))
        ss_error_total += ss_err
        if only is not None and eff != only:
            continue
        df1, df2 = k, k * (n - 1)
        f = (ss_eff / df1) / (ss_err / df2) if ss_err > 0 else np.inf
        if k > 1:
            s = np.cov(z, rowvar=False)
            eps = float(np.trace(s) ** 2 / (k * np.trace(s @ s)))
            eps = min(1.0, max(1.0 / k, eps))
        else:
            eps = 1.0
        rows.append(
            {
                "effect": ":".join(eff),
                "order": len(eff),
                "df1": df1,
                "df2": df2,
                "F": f,
                "gg_epsilon": eps,
                "p_gg": float(stats.f.sf(f, eps * df1, eps * df2)),
                "ss_effect": ss_eff,
                "ss_error": ss_err,
            }
        )
    out = pd.DataFrame(rows)
    out["ges"] = out.ss_effect / (out.ss_effect + ss_subjects + ss_error_total)
    return out


def anova_table(y: np.ndarray, names=FACTOR_NAMES) -> pd.DataFrame:
    """RM-ANOVA with BH-FDR over the main effects and two-way interactions (confirmatory)."""
    out = rm_anova(y, names)
    conf = out.order <= 2
    out["confirmatory"] = conf
    out["p_bh"] = np.nan
    out.loc[conf, "p_bh"] = fdr_bh(out.loc[conf, "p_gg"].to_numpy())
    return out


def _aligned(y: np.ndarray, effect: tuple[str, ...], names) -> np.ndarray:
    cell = y.mean(axis=0)
    resid = y - cell
    axes = {name: i for i, name in enumerate(names)}
    est = np.zeros_like(cell)
    for k in range(len(effect) + 1):
        for subset in itertools.combinations(effect, k):
            others = tuple(axes[n] for n in names if n not in subset)
            marg = cell.mean(axis=others, keepdims=True) if others else cell
            est = est + (-1) ** (len(effect) - k) * marg
    return resid + est


def art_anova(y: np.ndarray, names=FACTOR_NAMES) -> pd.DataFrame:
    """Aligned rank transform ANOVA: align and rank for each effect, then RM-ANOVA on ranks."""
    from scipy.stats import rankdata

    rows = []
    for eff in effects(names):
        ranks = rankdata(_aligned(y, eff, names), method="average").reshape(y.shape)
        rows.append(rm_anova(ranks, names, only=eff))
    out = pd.concat(rows, ignore_index=True)
    conf = out.order <= 2
    out["p_bh"] = np.nan
    out.loc[conf, "p_bh"] = fdr_bh(out.loc[conf, "p_gg"].to_numpy())
    return out.drop(columns=["ges"])


# --- planned contrasts --------------------------------------------------------------------------


def _marginal(y: np.ndarray, levels: dict, **fixed) -> np.ndarray:
    """Per-subject mean over all cells with the given factor levels (others averaged)."""
    sel = [slice(None)]
    for name in FACTOR_NAMES:
        if name in fixed:
            sel.append(levels[name].index(fixed[name]))
        else:
            sel.append(slice(None))
    sub = y[tuple(sel)]
    return sub.reshape(len(y), -1).mean(axis=1)


PLANNED = [
    ("P1", "B8 - B1", "does band power add to the waveform"),
    ("P2", "B6 - best single band (leave-one-subject-out)", "does combining bands help"),
    ("P3", "B1 - B2", "waveform vs theta envelope"),
    ("P4", "W1 - W2", "does adding the late window help"),
    ("P5", "W4 - W1", "does adding 0-200 ms hurt"),
    ("P6", "N4 - N2", "spatial whitening vs per-feature standardization"),
    ("P7", "N3 - N2", "does per-trial normalization hurt"),
    ("P8", "C1 - C2", "all channels vs the N400 region"),
    ("P9", "C2 - C3", "N400 region vs frontal"),
    ("P10", "C2 - C4", "N400 region vs posterior"),
    ("P11", "(C1 - C5)|W3 - (C1 - C5)|W2", "lateralized signal in the late window"),
]


def loso_best_band(bands: np.ndarray) -> np.ndarray:
    """For each subject, the column with the highest mean over the other subjects."""
    n = len(bands)
    total = bands.sum(axis=0)
    return np.array([int(np.argmax((total - bands[i]) / (n - 1))) for i in range(n)])


def contrast_differences(
    scores: pd.DataFrame, model: str
) -> tuple[list[str], dict[str, np.ndarray], dict[str, int]]:
    """Per-subject differences of P1-P11 (subjects, {code: differences}, P2 band counts)."""
    levels = levels_for(model)
    subjects, y = to_array(scores[scores.model == model], levels)
    m = lambda **kw: _marginal(y, levels, **kw)  # noqa: E731
    n_ref = "N1" if model == "lda" else "N2"  # LDA: N2 is identical to N1

    single = ["B2", "B3", "B4", "B5"]
    band_means = np.column_stack([m(B=b) for b in single])
    pick = loso_best_band(band_means)
    best = band_means[np.arange(len(y)), pick]

    diffs = {
        "P1": m(B="B8") - m(B="B1"),
        "P2": m(B="B6") - best,
        "P3": m(B="B1") - m(B="B2"),
        "P4": m(W="W1") - m(W="W2"),
        "P5": m(W="W4") - m(W="W1"),
        "P6": m(N="N4") - m(N=n_ref),
        "P7": m(N="N3") - m(N=n_ref),
        "P8": m(C="C1") - m(C="C2"),
        "P9": m(C="C2") - m(C="C3"),
        "P10": m(C="C2") - m(C="C4"),
        "P11": (m(C="C1", W="W3") - m(C="C5", W="W3")) - (m(C="C1", W="W2") - m(C="C5", W="W2")),
    }
    counts = {single[i]: int((pick == i).sum()) for i in range(len(single))}
    return subjects, diffs, counts


def planned_contrasts(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    _, diffs, counts = contrast_differences(scores, model)
    rows = []
    for code, label, question in PLANNED:
        s = summarize(diffs[code])
        rows.append(
            {"model": model, "contrast": code, "definition": label, "question": question, **s}
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["p_wilcoxon_holm"] = holm(out.p_wilcoxon.fillna(1.0).to_numpy())
    out["note"] = ""
    out.loc[out.contrast == "P2", "note"] = "best band chosen: " + json.dumps(counts)
    if model == "lda":
        out.loc[out.contrast.isin(["P6", "P7"]), "note"] = "N2 replaced by the identical N1"
    return out


# --- exploratory pairwise comparisons (inputs for scripts/pairwise.R) -----------------------------


def export_marginal_tables(scores: pd.DataFrame, model: str, out_dir: Path) -> list[Path]:
    """Subject-level marginal means for every factor and factor pair (long CSV files)."""
    levels = levels_for(model)
    subjects, y = to_array(scores[scores.model == model], levels)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k in (1, 2):
        for combo in itertools.combinations(FACTOR_NAMES, k):
            rows = []
            for lv in itertools.product(*[levels[f] for f in combo]):
                vals = _marginal(y, levels, **dict(zip(combo, lv, strict=True)))
                for s, v in zip(subjects, vals, strict=True):
                    rows.append({"subject": s, **dict(zip(combo, lv, strict=True)), "acc": v})
            path = out_dir / f"{model}_{'_'.join(combo)}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            written.append(path)
    return written


# --- time-resolved curves -----------------------------------------------------------------------


def timecourse_tests(
    tc: pd.DataFrame, seed: int = 4242, n_permutations: int = 1024
) -> pd.DataFrame:
    """Group curve per channel set with TFCE one-sided cluster permutation p-values."""
    from mne.stats import permutation_cluster_1samp_test
    from scipy import stats

    per_subject = tc.groupby(["channels", "bin_start", "subject"], as_index=False).bacc.mean()
    rows = []
    for code, g in per_subject.groupby("channels"):
        wide = g.pivot(index="subject", columns="bin_start", values="bacc")
        x = wide.to_numpy()
        _, clusters, cluster_pv, _ = permutation_cluster_1samp_test(
            x - 0.5,
            threshold={"start": 0, "step": 0.2},
            n_permutations=n_permutations,
            tail=1,
            seed=seed,
            out_type="indices",
            verbose=False,
        )
        p = np.ones(x.shape[1])
        for c, pv in zip(clusters, cluster_pv, strict=True):
            p[np.asarray(c[0])] = pv
        n = len(x)
        half = stats.t.ppf(0.975, n - 1) * x.std(axis=0, ddof=1) / np.sqrt(n)
        for j, start in enumerate(wide.columns):
            rows.append(
                {
                    "channels": code,
                    "bin_start": start,
                    "mean_bacc": x[:, j].mean(),
                    "ci_low": x[:, j].mean() - half[j],
                    "ci_high": x[:, j].mean() + half[j],
                    "p_tfce": p[j],
                }
            )
    return pd.DataFrame(rows)


# --- stage reports ------------------------------------------------------------------------------


def gate_report(scores: pd.DataFrame, permutations: pd.DataFrame | None = None) -> dict:
    """Baseline: group tests (the gate) and per-subject results."""
    out = {}
    for model, g in scores.groupby("model"):
        out[model] = summarize(g.bacc.to_numpy() - 0.5, alternative="greater")
        out[model]["mean_bacc"] = float(g.bacc.mean())
    if permutations is not None and len(permutations):
        sig = permutations.p < 0.05
        out["per_subject"] = {
            "n_subjects": int(len(permutations)),
            "n_significant_p05": int(sig.sum()),
            "expected_by_chance": 0.05 * len(permutations),
        }
    return out


def pilot_report(folds: pd.DataFrame, c_grid) -> pd.DataFrame:
    """Per model x configuration: accuracy, time per fold, and how often C hits a grid edge."""
    lo, hi = min(c_grid), max(c_grid)
    rows = []
    for (model, config), g in folds.groupby(["model", "config"]):
        row = {
            "model": model,
            "config": config,
            "mean_bacc": g.groupby("subject").bacc.mean().mean(),
            "seconds_per_fold": g.seconds.mean(),
            "seconds_per_subject": g.groupby("subject").seconds.sum().mean(),
        }
        if model == "lr":
            row["frac_C_min"] = float((g.C_selected == lo).mean())
            row["frac_C_max"] = float((g.C_selected == hi).mean())
            row["C_edge_warning"] = bool(row["frac_C_min"] > 0.2 or row["frac_C_max"] > 0.2)
        rows.append(row)
    return pd.DataFrame(rows)


def behaviour_table(deriv_root: Path, subjects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-subject behaviour and pooled reaction-time quantiles (section 6.5 step 1)."""
    from .events import behaviour_summary

    per_subject, rts = [], []
    for s in subjects:
        path = deriv_root / s / "trials.tsv"
        if not path.exists():
            continue
        trials = pd.read_csv(path, sep="\t")
        per_subject.append({"subject": s, **behaviour_summary(trials)})
        ok = trials[trials.correct == 1.0]
        rts.append(ok[["condition", "rt_ms"]])
    pooled = pd.concat(rts) if rts else pd.DataFrame(columns=["condition", "rt_ms"])
    q = [0.05, 0.10, 0.25, 0.50, 0.75]
    quant = pooled.groupby("condition").rt_ms.quantile(q).unstack()
    quant.loc["all"] = pooled.rt_ms.quantile(q)
    return pd.DataFrame(per_subject), quant.reset_index()
