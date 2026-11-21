"""Sensitivity analysis: is the waveform still not helped by a stronger time-frequency
representation? (engineering guide section 13.10)

Inputs per task: the main logistic-regression results of ``B1-W1-N2-C1`` and ``B6-W1-N2-C1``
(``<task results>/reports/full/subject_scores.csv``) and the ``tfr`` stage (``decode --stage
tfr``, ``<task results>/tfr_sensitivity/tfr/``), run on the same outer folds. Outputs in
``<results>/tfr_sensitivity/``:

1. ``means.csv``: mean balanced accuracy of B1, B6 and T1-T5 per task;
2. ``per_task.csv``: the six comparisons S1-S6 in each task (paired t tests, Holm over the six
   within a task);
3. ``joint.csv``: the joint three-task test of each comparison (section 13.5: equal weights,
   Welch df, Holm over the six, heterogeneity, category);
4. ``stacking_weights.csv``: T5 meta-model weights of B1 and T1 over the outer folds, and the
   share of T1, ``w_T1 / (|w_B1| + |w_T1|)``;
5. the pre-specified wording that applies (printed and in ``conclusion.txt``).

    python scripts/tfr_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from n400fp.collect import collect_folds
from n400fp.joint import CATEGORIES, TASKS, classify, joint_test
from n400fp.paths import results_root, task_results_dir
from n400fp.run_subject import subject_means
from n400fp.stats import holm, summarize
from n400fp.study import TFR_FIXED, TFR_LEVELS

MAIN = {"B1": "B1-W1-N2-C1", "B6": "B6-W1-N2-C1"}
LEVELS = ("B1", "B6", *TFR_LEVELS)
# code, minuend, subtrahend, primary
COMPARISONS = (
    ("S1", "T4", "B1", True),
    ("S2", "T5", "B1", True),
    ("S3", "T1", "B6", False),
    ("S4", "T1", "B1", False),
    ("S5", "T2", "T1", False),
    ("S6", "T3", "B1", False),
)
RELIABLE = CATEGORIES[:2]  # "general", "same direction, size varies"


def load_main(task: str) -> pd.DataFrame:
    """Subject x {B1, B6} accuracies of the main logistic regression, plus ``n_folds``."""
    scores = pd.read_csv(task_results_dir(task) / "reports" / "full" / "subject_scores.csv")
    scores = scores[(scores.model == "lr") & scores.config.isin(MAIN.values())]
    wide = scores.pivot(index="subject", columns="config", values="bacc")
    wide = wide.rename(columns={v: k for k, v in MAIN.items()})[list(MAIN)]
    if "n_folds" in scores:
        wide["n_folds"] = scores.groupby("subject").n_folds.max()
    return wide


def load_tfr(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject x {T1..T5} accuracies of the ``tfr`` stage, and its per-fold rows."""
    folds = collect_folds("tfr", task_results_dir(task) / "tfr_sensitivity")
    folds = folds[folds.model == "lr"]
    means = subject_means(folds)
    wide = means.pivot(index="subject", columns="B", values="bacc")
    wide["n_folds"] = means.groupby("subject").n_folds.min()
    return wide, folds


def check_complete(main: pd.DataFrame, tfr: pd.DataFrame, task: str) -> None:
    if set(tfr.index) != set(main.index):
        missing = sorted(set(main.index) ^ set(tfr.index))
        raise SystemExit(f"{task}: subjects differ between the main and tfr results: {missing}")
    absent = [b for b in TFR_LEVELS if b not in tfr or tfr[b].isna().any()]
    if absent:
        raise SystemExit(f"{task}: tfr representations missing for some subjects: {absent}")
    if "n_folds" in main and not (tfr.n_folds == main.n_folds.reindex(tfr.index)).all():
        raise SystemExit(f"{task}: fold counts differ from the main run")


def combine(main: pd.DataFrame, tfr: pd.DataFrame) -> pd.DataFrame:
    return main[list(MAIN)].join(tfr[list(TFR_LEVELS)], how="inner")[list(LEVELS)]


def differences(acc: pd.DataFrame) -> dict[str, pd.Series]:
    return {code: acc[a] - acc[b] for code, a, b, _ in COMPARISONS}


def per_task_table(acc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        diffs = differences(acc[task])
        part = []
        for code, a, b, primary in COMPARISONS:
            part.append(
                {
                    "task": task,
                    "comparison": code,
                    "definition": f"{a} - {b}",
                    "primary": primary,
                    **summarize(diffs[code].to_numpy()),
                }  # fmt: skip
            )
        p_holm = holm([r["p"] for r in part])
        rows += [{**r, "p_holm": p} for r, p in zip(part, p_holm, strict=True)]
    return pd.DataFrame(rows)


def joint_table(acc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    diffs = {task: differences(acc[task]) for task in TASKS}
    rows = []
    for code, a, b, primary in COMPARISONS:
        row = {"comparison": code, "definition": f"{a} - {b}", "primary": primary}
        row.update(joint_test(*(diffs[task][code] for task in TASKS)))
        for task in TASKS:
            row[f"{task}_mean"] = diffs[task][code].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["heterogeneity_p_holm"] = holm(out.heterogeneity_p.to_numpy())
    out["category"] = [
        classify(r.p_holm, [getattr(r, f"{t}_mean") for t in TASKS], r.heterogeneity_p_holm)
        for r in out.itertuples()
    ]
    positive = (out.p_holm < 0.05) & (out.estimate > 0)
    out["reliable_positive"] = positive & out.category.isin(RELIABLE)
    out["positive_not_reliable"] = positive & ~out.reliable_positive
    return out


def conclusion(joint: pd.DataFrame) -> str:
    """The wording pre-specified in guide 13.10 for the observed outcome."""
    pos = joint.set_index("comparison").reliable_positive
    if pos["S1"]:
        text = (
            "S1 has a reliable positive effect: the statement that adding power hurts must be "
            "revised. Report to the researcher before changing the documents."
        )
    elif pos["S2"]:
        text = (
            "Only S2 has a reliable positive effect: power carries a little information beyond "
            "the waveform, but concatenating it does not help or hurts."
        )
    else:
        text = (
            "Neither S1 nor S2 has a reliable positive effect: the conclusion stands and is "
            "strengthened. Even fine-grained power including delta, combined by stacking, adds "
            "nothing beyond the waveform."
        )
    flagged = joint.set_index("comparison").positive_not_reliable
    for code in ("S1", "S2"):
        if flagged[code]:
            text += (
                f" Caveat: {code} has a significant positive joint effect, but its tasks disagree "
                "(category not 'general' or 'same direction'); report it to the researcher."
            )
    if not pos["S3"]:
        text += (
            " S3 is not reliably positive: this analysis did not produce a stronger power "
            "representation than the four broad bands; say so when reporting S1 and S2."
        )
    return text


def stacking_weights(folds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        t5 = folds[task][folds[task].B == "T5"]
        w_wave, w_tfr = t5.w_part0, t5.w_part1
        # share of the weight on T1; the weights are on each part's decision-value scale
        share = w_tfr / (w_wave.abs() + w_tfr.abs())
        rows.append(
            {
                "task": task,
                "n_folds": len(t5),
                "w_B1_median": w_wave.median(),
                "w_T1_median": w_tfr.median(),
                "frac_w_B1_positive": float((w_wave > 0).mean()),
                "frac_w_T1_positive": float((w_tfr > 0).mean()),
                "share_T1_median": float(np.nanmedian(share)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    out = results_root() / "tfr_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    acc, folds = {}, {}
    for task in TASKS:
        main_acc = load_main(task)
        tfr_acc, folds[task] = load_tfr(task)
        check_complete(main_acc, tfr_acc, task)
        acc[task] = combine(main_acc, tfr_acc)
    means = pd.DataFrame({task: acc[task].mean() for task in TASKS}).T
    means.index.name = "task"
    per_task = per_task_table(acc)
    joint = joint_table(acc)
    weights = stacking_weights(folds)
    text = conclusion(joint)

    means.to_csv(out / "means.csv")
    per_task.to_csv(out / "per_task.csv", index=False)
    joint.to_csv(out / "joint.csv", index=False)
    weights.to_csv(out / "stacking_weights.csv", index=False)
    (out / "conclusion.txt").write_text(text + "\n", encoding="utf-8")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(f"== mean balanced accuracy (LR, {'-'.join(TFR_FIXED)})")
        print(means.round(4).to_string())
        print("== per task")
        cols = ["task", "comparison", "definition", "mean", "dz", "p", "p_holm"]
        print(per_task[cols].round(4).to_string(index=False))
        print("== joint test")
        cols = ["comparison", "definition", "estimate", "ci_low", "ci_high", "dz", "p_holm",
                "heterogeneity_p_holm", "category", "reliable_positive"]  # fmt: skip
        print(joint[cols].round(4).to_string(index=False))
        print("== T5 stacking weights")
        print(weights.round(4).to_string(index=False))
    print("== conclusion\n" + text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
