"""Tables of the supplementary material (paper/NeuroImage/supplement/tables/), from local results.

Every number in the supplementary tables comes from this script, so that nothing is copied by
hand. Two analyses are computed here rather than read: the joint test without the partly flat
recording (S7) and the EEGNet versus logistic-regression means (S5).

    python scripts/paper_supplement.py
"""

from __future__ import annotations

import json
import re

import pandas as pd

from n400fp.joint import TASKS, joint_contrasts
from n400fp.paths import REPO_ROOT, results_root, task_results_dir

OUT = REPO_ROOT / "paper" / "NeuroImage" / "supplement" / "tables"
LABELS = {"erpcore": "Word pairs", "scene": "Scenes", "sentence": "Sentences", "pursue": "PURSUE"}
CATEGORY = {
    "general": "G",
    "same direction, size varies": "C",
    "task-dependent": "T",
    "no reliable general effect": "N",
}
EFFECTS = ["B", "W", "N", "C", "B:W", "B:N", "B:C", "W:N", "W:C", "N:C"]
BAND_NAMES = {"B2": "theta", "B3": "alpha", "B4": "beta", "B5": "low gamma"}
TASK_NOUNS = {"erpcore": "word-pair", "scene": "scene", "sentence": "sentence"}


# --- formatting ------------------------------------------------------------------------------


def pp(x: float, digits: int = 2) -> str:
    """Accuracy difference in percentage points, signed (unsigned when it rounds to zero)."""
    text = f"{100 * x:+.{digits}f}"
    return f"${text[1:] if float(text) == 0 else text}$"


def pval(p: float) -> str:
    if p < 0.0001:
        return "$<0.0001$"
    return f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def num(x: float, digits: int = 3) -> str:
    text = f"{x:.{digits}f}"
    return f"${text.lstrip('-') if float(text) == 0 else text}$"


def write(
    name: str,
    caption: str,
    label: str,
    colspec: str,
    header: str,
    rows: list[str],
    size: str = "\\footnotesize",
) -> None:
    """A booktabs table, shrunk to the text width only when it is wider (never enlarged)."""
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n{size}\n"
        f"\\begin{{adjustbox}}{{max width=\\textwidth}}\n\\begin{{tabular}}{{{colspec}}}\n"
        f"\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\end{adjustbox}\n\\end{table}\n"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8")
    print(f"wrote {OUT / name}.tex")


def scores(task: str, root=None) -> pd.DataFrame:
    root = root or task_results_dir(task)
    return pd.read_csv(root / "reports" / "full" / "subject_scores.csv")


# --- S1: preprocessing -----------------------------------------------------------------------


def mean_range(x: pd.Series, digits: int = 1, range_digits: int = 0) -> str:
    """Mean (minimum--maximum); "--" when the step was not applied (all zero)."""
    if not x.any():
        return "--"
    lo, hi = f"{x.min():.{range_digits}f}", f"{x.max():.{range_digits}f}"
    return f"{x.mean():.{digits}f} ({lo}--{hi})"


def table_preprocessing() -> None:
    folders = {
        "erpcore": results_root(),
        "scene": results_root() / "scene",
        "sentence": results_root() / "sentence",
        "pursue": results_root() / "pursue",
    }
    rows = []
    for task, folder in folders.items():
        s = pd.read_csv(folder / "reports" / "preprocessing" / "preproc_summary.csv")
        kept, total = s.n_kept.sum(), s.n_targets.sum()
        rejected = int(s.get("n_dropped_threshold", pd.Series(0)).sum())
        flat = int(s.get("n_dropped_flat", pd.Series(0)).sum())
        rows.append(
            " & ".join(
                [
                    LABELS[task],
                    str(len(s)),
                    f"{kept} / {total} ({100 * kept / total:.1f}\\%)",
                    f"{s.n_kept.min()}--{s.n_kept.max()}",
                    mean_range(s.ica_eog_n),
                    mean_range(s.ica_muscle_n),
                    mean_range(s.interpolated_fraction, 3, 3),
                    str(int(s.n_dropped_unrepairable.sum())),
                    str(rejected),
                    str(flat),
                ]
            )
            + " \\\\"
        )
    header = (
        "Dataset & $n$ & Trials kept & Per participant & Ocular ICs & Muscle ICs & "
        "Interpolated & No good channel & Threshold & Flat"
    )
    write(
        "s1_preprocessing",
        "Preprocessing summary (main analysis). Ocular and muscle independent "
        "components (ICs) removed and fraction of interpolated channel-epochs: mean (range). "
        "The last three columns count rejected epochs: without any good channel, above the global "
        "peak-to-peak threshold, and flat (sentence task only); --, step not applied. PURSUE "
        "includes the participant excluded after preprocessing.",
        "tab:s_preproc",
        "@{}lrlrlllrrr@{}",
        header,
        rows,
    )


# --- S2: preprocessing sensitivity -----------------------------------------------------------


def table_preproc_sensitivity() -> None:
    folder = results_root() / "preproc_sensitivity"
    summary = pd.read_csv(folder / "summary.csv")
    rows = [
        " & ".join(
            [
                LABELS[r.task],
                r.model.upper(),
                f"\\texttt{{{r.baseline_config}}}",
                num(r.baseline_main),
                num(r.baseline_prespecified),
                pp(r.baseline_diff),
                pval(r.baseline_diff_p),
                num(r.config_means_r, 3),
            ]
        )
        + " \\\\"
        for r in summary.itertuples()
    ]
    header = (
        "Task & Model & Baseline & Main & Original & Difference (pp) & $p$ & $r$ (configurations)"
    )
    write(
        "s2_preproc_baseline",
        "Baseline accuracy with the main and the originally specified "
        "preprocessing, and correlation of the configuration means between the two.",
        "tab:s_preproc_baseline",
        "@{}llllllll@{}",
        header,
        rows,
    )

    joint = pd.read_csv(folder / "joint.csv")
    rows = []
    for r in joint.itertuples():
        changed = " $\\leftarrow$" if r.category_changed else ""
        rows.append(
            " & ".join(
                [
                    r.model.upper(),
                    r.contrast,
                    pp(r.estimate_main),
                    pval(r.p_holm_main),
                    CATEGORY[r.category_main],
                    pp(r.estimate_prespecified),
                    pval(r.p_holm_prespecified),
                    CATEGORY[r.category_prespecified] + changed,
                ]
            )
            + " \\\\"
        )
    header = (
        "Model & Contrast & Main (pp) & $p_{\\text{Holm}}$ & Category & Original (pp) & "
        "$p_{\\text{Holm}}$ & Category"
    )
    write(
        "s2_preproc_joint",
        "Joint test with the main and the originally specified preprocessing "
        "for the scene and sentence tasks (ERP CORE unchanged). Categories as in Table~3 of the "
        "main text; the arrow marks the one changed category.",
        "tab:s_preproc_joint",
        "@{}llrrcrrc@{}",
        header,
        rows,
    )


# --- S3: analyses within each dataset --------------------------------------------------------


def table_planned(model: str) -> None:
    rows = []
    for task in TASKS:
        planned = pd.read_csv(
            task_results_dir(task) / "reports" / "full" / f"planned_contrasts_{model}.csv"
        )
        for i, r in enumerate(planned.itertuples()):
            name = LABELS[task] if i == 0 else ""
            rows.append(
                " & ".join(
                    [
                        name,
                        r.contrast,
                        f"{pp(r.mean)} ({pp(r.ci_low)}, {pp(r.ci_high)})",
                        num(r.dz, 2),
                        pval(r.p_holm),
                        pval(r.p_wilcoxon_holm),
                    ]
                )
                + " \\\\"
            )
        rows.append("\\addlinespace")
    header = (
        "Task & Contrast & Difference, pp (95\\% CI) & $d_z$ & $p_{\\text{Holm}}$ & "
        "$p_{\\text{Holm}}$ (Wilcoxon)"
    )
    name = "logistic regression" if model == "lr" else "shrinkage LDA (N1 in place of N2)"
    write(
        f"s3_planned_{model}",
        f"Planned contrasts within each dataset, {name}; Holm correction "
        "across the eleven contrasts of a dataset.",
        f"tab:s_planned_{model}",
        "@{}llllll@{}",
        header,
        rows[:-1],
    )


def table_anova() -> None:
    rows = []
    frames = {}
    for task in TASKS:
        folder = task_results_dir(task) / "reports" / "full"
        rm = pd.read_csv(folder / "rm_anova_lr.csv").set_index("effect")
        art = pd.read_csv(folder / "art_anova_lr.csv").set_index("effect")
        frames[task] = (rm, art)
    for effect in EFFECTS:
        cells = [effect.replace(":", " $\\times$ ")]
        for task in TASKS:
            rm, art = frames[task]
            r = rm.loc[effect]
            cells += [f"{r.F:.1f}", pval(r.p_bh), f"{r.ges:.3f}", pval(art.loc[effect].p_bh)]
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join(["Effect"] + [f"\\multicolumn{{4}}{{c}}{{{LABELS[t]}}}" for t in TASKS])
    header += " \\\\\n" + " & ".join(
        [""] + ["$F$ & $p_{\\text{BH}}$ & $\\eta^2_G$ & ART $p_{\\text{BH}}$"] * 3
    )
    write(
        "s3_anova",
        "Four-way repeated-measures analysis of variance within each dataset "
        "(logistic regression; Greenhouse--Geisser corrected; Benjamini--Hochberg correction over "
        "the four main effects and six two-way interactions) and the aligned-rank-transform check.",
        "tab:s_anova",
        "@{}l" + "rrrr" * 3 + "@{}",
        header,
        rows,
    )


def table_representation_window() -> None:
    """Marginal accuracy of each representation in each time window (LR, per task)."""
    reps = [f"B{i}" for i in range(1, 9)]
    windows = ["W1", "W2", "W3", "W4"]
    means = {}
    for task in TASKS:
        path = task_results_dir(task) / "reports" / "full" / "marginal_tables" / "lr_B_W.csv"
        means[task] = pd.read_csv(path).groupby(["B", "W"])["acc"].mean()
    rows = []
    for rep in reps:
        cells = [rep] + [num(means[task].loc[(rep, w)]) for task in TASKS for w in windows]
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join(
        ["Representation"] + [f"\\multicolumn{{4}}{{c}}{{{LABELS[t]}}}" for t in TASKS]
    )
    header += " \\\\\n" + " & ".join([""] + [" & ".join(windows)] * len(TASKS))
    write(
        "s3_representation_window",
        "Mean balanced accuracy of each representation in each time window (logistic "
        "regression; each participant's accuracy averaged over normalization and channel set). "
        "W1, 200--800\\,ms; W2, 300--500\\,ms; W3, 500--800\\,ms; W4, 0--800\\,ms. "
        "B1, waveform; B2--B5, theta, alpha, beta and low-gamma envelopes; B6, B2--B5 "
        "concatenated; B7, relative band amplitudes; B8, B1 and B2--B5 concatenated.",
        "tab:s_rep_window",
        "@{}l" + "rrrr" * 3 + "@{}",
        header,
        rows,
    )


# --- S4: joint test with LDA -----------------------------------------------------------------


def table_joint_lda() -> None:
    joint = pd.read_csv(results_root() / "joint" / "joint_contrasts_lda.csv")
    rows = [
        " & ".join(
            [
                r.contrast,
                f"{pp(r.estimate)} ({pp(r.ci_low)}, {pp(r.ci_high)})",
                num(r.dz, 2),
                pval(r.p_holm),
                pp(r.erpcore_mean),
                pp(r.scene_mean),
                pp(r.sentence_mean),
                pval(r.heterogeneity_p_holm),
                CATEGORY[r.category],
            ]
        )
        + " \\\\"
        for r in joint.itertuples()
    ]
    header = (
        "Contrast & Joint, pp (95\\% CI) & $d_z$ & $p_{\\text{Holm}}$ & Word pairs & Scenes & "
        "Sentences & $p_{\\text{het}}$ & Category"
    )
    write(
        "s4_joint_lda",
        "Joint test across the three tasks with shrinkage LDA (N1 in place of N2 "
        "in P6 and P7). Categories as in Table~3 of the main text.",
        "tab:s_joint_lda",
        "@{}llrrrrrrc@{}",
        header,
        rows,
    )


# --- S5: EEGNet ------------------------------------------------------------------------------


def eegnet_means(task: str) -> pd.Series:
    folder = task_results_dir(task) / "eegnet"
    folds = pd.concat(pd.read_parquet(p) for p in sorted(folder.glob("sub-*.parquet")))
    per_subject = folds.groupby(["config", "subject"]).bacc.mean()
    return per_subject.groupby("config").mean()


def table_eegnet() -> None:
    rows = []
    for task in TASKS:
        eeg = eegnet_means(task)
        s = scores(task)
        lr = s[s.model == "lr"].groupby("config").bacc.mean()
        for i, config in enumerate(eeg.sort_values(ascending=False).index):
            name = LABELS[task] if i == 0 else ""
            rows.append(
                f"{name} & \\texttt{{{config}}} & {num(eeg[config])} & {num(lr[config])} & "
                f"{pp(eeg[config] - lr[config])} \\\\"
            )
        rows.append("\\addlinespace")
    header = "Task & Configuration & EEGNet & Logistic regression & Difference (pp)"
    write(
        "s5_eegnet",
        "EEGNet compared with logistic regression on the baseline and the ten best "
        "and three worst time-series configurations of each task (mean balanced accuracy).",
        "tab:s_eegnet",
        "@{}llrrr@{}",
        header,
        rows[:-1],
    )


# --- S6: regularization grid -----------------------------------------------------------------


def table_cgrid() -> None:
    summary = pd.read_csv(results_root() / "c_grid_sensitivity" / "summary.csv")
    rows = [
        " & ".join(
            [
                LABELS[r.task],
                num(r.baseline_study),
                num(r.baseline_extended),
                pp(r.baseline_diff),
                num(r.all_configs_study),
                num(r.all_configs_extended),
                num(r.config_means_r),
                f"{100 * r.frac_below_study_grid:.1f}\\%",
                f"{100 * r.frac_above_study_grid:.1f}\\%",
            ]
        )
        + " \\\\"
        for r in summary.itertuples()
    ]
    header = (
        " & \\multicolumn{3}{c}{Baseline} & \\multicolumn{3}{c}{All 640 configurations} & "
        "\\multicolumn{2}{c}{Outside study grid} \\\\\n"
        "Task & Study & Extended & Difference (pp) & Study & Extended & $r$ & "
        "$C < 10^{-3}$ & $C > 10^{2}$"
    )
    write(
        "s6_cgrid",
        "Logistic regression with the study grid ($10^{-3}$ to $10^2$) and the "
        "extended grid ($10^{-5}$ to $10^4$) on the same folds: mean balanced accuracy at baseline "
        "and over all 640 configurations, and the correlation $r$ of the 640 configuration means "
        "between the two grids. The last two columns give the fraction of outer folds in which the "
        "extended grid selected a value outside the study grid.",
        "tab:s_cgrid",
        "@{}lrrrrrrrr@{}",
        header,
        rows,
    )


# --- S7: the partly flat recording -----------------------------------------------------------


def joint_without(exclude: dict[str, set[str]]) -> pd.DataFrame:
    data = {}
    for task in TASKS:
        s = scores(task)
        data[task] = s[~s.subject.isin(exclude.get(task, set()))]
    return joint_contrasts(data, "lr").set_index("contrast")


def table_sub017() -> None:
    variants = {
        "All participants": {},
        "Without sub-017 (scene and sentence)": {"scene": {"sub-017"}, "sentence": {"sub-017"}},
        "Also without ERP CORE sub-029, sub-040": {
            "scene": {"sub-017"},
            "sentence": {"sub-017"},
            "erpcore": {"sub-029", "sub-040"},
        },
    }
    results = {name: joint_without(ex) for name, ex in variants.items()}
    rows = []
    for name, j in results.items():
        cells = [name]
        for c in ("P1", "P2", "P3", "P7"):  # the contrasts significant in the main analysis
            r = j.loc[c]
            cells.append(f"{pp(r.estimate)}, {pval(r.p_holm)}, {CATEGORY[r.category]}")
        rows.append(" & ".join(cells) + " \\\\")
    header = "Sample & P1 & P2 & P3 & P7"
    write(
        "s7_sub017",
        "Joint test (logistic regression) without the partly flat sentence recording "
        "(Draschkow participant sub-017, removed from both of that participant's tasks) and, in "
        "addition, without ERP CORE sub-029 (40 epochs without a good channel) and sub-040 "
        "(excluded in the original ERP CORE analyses). Each cell: "
        "estimate (pp), $p_{\\text{Holm}}$, category.",
        "tab:s_sub017",
        "@{}lcccc@{}",
        header,
        rows,
    )

    # Completes the sentence "the best single band chosen by the ... rule was ..." in S3.
    phrases = []
    for task in TASKS:
        planned = pd.read_csv(
            task_results_dir(task) / "reports" / "full" / "planned_contrasts_lr.csv"
        )
        note = planned.set_index("contrast").loc["P2", "note"]
        chosen = {b: n for b, n in json.loads(note.split(": ", 1)[1]).items() if n}
        bands = [f"{BAND_NAMES[b]} ({b}) for {n}" for b, n in chosen.items()]
        if len(bands) == 1:
            bands = [bands[0].replace(" for ", " for all ") + " participants"]
        phrases.append(f"{' and '.join(bands)} in the {TASK_NOUNS[task]} task")
    sentence = f"{', '.join(phrases[:-1])}, and {phrases[-1]}"
    (OUT / "s7_best_band.tex").write_text(sentence + "\n", encoding="utf-8")


# --- S8: exploratory analyses ----------------------------------------------------------------

MEASURES = [
    ("wave_n400_uV", "Waveform, 300--500\\,ms (\\textmu V)", 2),
    ("wave_late_uV", "Waveform, 500--800\\,ms (\\textmu V)", 2),
    ("theta_n400_log10", "Theta envelope, 300--500\\,ms ($\\log_{10}$)", 3),
    ("alpha_n400_log10", "Alpha envelope, 300--500\\,ms", 3),
    ("beta_n400_log10", "Beta envelope, 300--500\\,ms", 3),
    ("gamma_n400_log10", "Low-gamma envelope, 300--500\\,ms", 3),
    ("theta_late_log10", "Theta envelope, 500--800\\,ms", 3),
    ("alpha_late_log10", "Alpha envelope, 500--800\\,ms", 3),
    ("beta_late_log10", "Beta envelope, 500--800\\,ms", 3),
    ("gamma_late_log10", "Low-gamma envelope, 500--800\\,ms", 3),
]


def table_condition_effects() -> None:
    frames = {
        t: pd.read_csv(
            task_results_dir(t) / "reports" / "full" / "exploratory" / "condition_effects_group.csv"
        ).set_index("measure")
        for t in TASKS
    }
    rows = []
    for key, name, digits in MEASURES:
        cells = [name]
        for task in TASKS:
            if key not in frames[task].index:  # envelopes of the Draschkow tasks are not local
                cells += ["--", "--"]
                continue
            r = frames[task].loc[key]
            cells += [f"${r['mean']:+.{digits}f}$".replace("+-", "-"), pval(r.p)]
        rows.append(" & ".join(cells) + " \\\\")
    header = " & ".join(["Measure"] + [f"\\multicolumn{{2}}{{c}}{{{LABELS[t]}}}" for t in TASKS])
    header += " \\\\\n" + " & ".join([""] + ["Effect & $p$"] * 3)
    write(
        "s8_condition_effects",
        "Exploratory condition differences (incongruent $-$ congruent) "
        "over the centro-parietal region: waveform in microvolts, band envelopes in $\\log_{10}$ "
        "amplitude (0.016 corresponds to about 4\\%). Uncorrected one-sample $t$-tests across "
        "participants. Envelope differences were computed for ERP CORE only.",
        "tab:s_condition",
        "@{}l" + "rr" * 3 + "@{}",
        header,
        rows,
    )


def pair_label(pair: str) -> str:
    """Variable pair as written in correlations.csv ("a ~ b") in the notation of the paper."""
    text = pair.replace(" ~ ", " vs.\\ ").replace("(uV)", "(\\textmu V)").replace("(d)", "($d$)")
    text = re.sub(r"\b(B\d-W\d-N\d-C\d)\b", r"\\texttt{\1}", text)
    return text[0].upper() + text[1:]


def table_correlations() -> None:
    rows = []
    for task in TASKS:
        c = pd.read_csv(
            task_results_dir(task) / "reports" / "full" / "exploratory" / "correlations.csv"
        )
        c = c[~c.pair.str.contains("RT effect")]  # reaction-time rows are not reported
        for i, r in enumerate(c.itertuples()):
            name = LABELS[task] if i == 0 else ""
            cells = [name, pair_label(r.pair), num(r.r, 2), pval(r.p), num(r.rho, 2)]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\addlinespace")
    header = "Task & Variables & Pearson $r$ & $p$ & Spearman $\\rho$"
    write(
        "s8_correlations",
        "Exploratory correlations across participants ($n = 40$ per task). N400 and late effect, "
        "incongruent $-$ congruent over the centro-parietal region at 300--500 and 500--800\\,ms, "
        "in \\textmu V or as a single-trial effect size $d$; LR, logistic regression.",
        "tab:s_correlations",
        "@{}lp{0.55\\textwidth}rrr@{}",
        header,
        rows[:-1],
    )


# --- S9: time-frequency analysis -------------------------------------------------------------


def table_tfr() -> None:
    folder = results_root() / "tfr_sensitivity"
    means = pd.read_csv(folder / "means.csv").set_index("task")
    weights = pd.read_csv(folder / "stacking_weights.csv").set_index("task")
    cols = ["B1", "B6", "T1", "T2", "T3", "T4", "T5"]
    rows = []
    for task in TASKS:
        w = weights.loc[task]
        cells = [LABELS[task]] + [num(means.loc[task, c]) for c in cols]
        cells += [f"{100 * w.frac_w_T1_positive:.0f}\\%", num(w.share_T1_median, 2)]
        rows.append(" & ".join(cells) + " \\\\")
    header = "Task & B1 & B6 & T1 & T2 & T3 & T4 & T5 & $w_{\\text{T1}} > 0$ & " "Share of T1"
    write(
        "s9_tfr",
        "Time--frequency sensitivity analysis: mean balanced accuracy of each "
        "representation (logistic regression; other dimensions at baseline), the fraction of outer "
        "folds in which the stacking meta-model gave wavelet power (T1) a positive weight, and the "
        "median share of T1, $|w_{\\text{T1}}| / (|w_{\\text{B1}}| + |w_{\\text{T1}}|)$.",
        "tab:s_tfr",
        "@{}l" + "r" * 9 + "@{}",
        header,
        rows,
    )


def main() -> None:
    table_preprocessing()
    table_preproc_sensitivity()
    table_planned("lr")
    table_planned("lda")
    table_anova()
    table_representation_window()
    table_joint_lda()
    table_eegnet()
    table_cgrid()
    table_sub017()
    table_condition_effects()
    table_correlations()
    table_tfr()
    corr = results_root() / "joint" / "scene_sentence_subject_correlation.json"
    print(f"scene-sentence baseline correlation: {json.loads(corr.read_text())}")


if __name__ == "__main__":
    main()
