"""Sensitivity analysis: do the logistic-regression results depend on the C grid?

The full stage of every task is rerun with the extended grid (1e-5 ... 1e4; ``decode --c-grid
extended``, results in ``<task results>/c_grid_extended/full/``) on the same outer folds. This
script compares it with the main run (pre-specified grid 1e-3 ... 100), for the logistic
regression only (LDA has no C):

1. ``summary.csv``: per task, the baseline and all-configuration mean accuracy under both
   grids, the correlation of the 640 configuration means, and how often the extended grid
   selects a C outside the study grid or at its own ends;
2. ``planned_<task>.csv``: the 11 planned contrasts of each task under both grids;
3. ``joint.csv``: the joint three-task test under both grids, and whether a category changed.

The main analysis stays the pre-specified one; this is a robustness check. Output in
``<results>/c_grid_sensitivity/``.

    python scripts/c_grid_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from n400fp.collect import collect_folds
from n400fp.config import C_GRIDS
from n400fp.joint import TASKS, joint_contrasts
from n400fp.paths import results_root, task_results_dir
from n400fp.run_subject import subject_means
from n400fp.stats import planned_contrasts
from n400fp.study import BASELINE, full_factorial

STUDY, EXTENDED = C_GRIDS["study"], C_GRIDS["extended"]
N_CONFIGS = len(full_factorial())


def load_main(task: str) -> pd.DataFrame:
    scores = pd.read_csv(task_results_dir(task) / "reports" / "full" / "subject_scores.csv")
    return scores[scores.model == "lr"].reset_index(drop=True)


def load_extended(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = collect_folds("full", task_results_dir(task) / "c_grid_extended")
    folds = folds[folds.model == "lr"]
    return subject_means(folds), folds


def check_complete(main: pd.DataFrame, ext: pd.DataFrame, task: str) -> None:
    """Same subjects, all 640 configurations each, same number of folds as the main run."""
    if set(ext.subject) != set(main.subject):
        missing = sorted(set(main.subject) - set(ext.subject))
        raise SystemExit(f"{task}: extended grid missing subjects {missing}")
    per_subject = ext.groupby("subject").config.nunique()
    if (per_subject != N_CONFIGS).any():
        short = per_subject[per_subject != N_CONFIGS].to_dict()
        raise SystemExit(f"{task}: incomplete subjects (configurations found): {short}")
    if "n_folds" in main and set(ext.n_folds) != set(main.n_folds):
        raise SystemExit(f"{task}: fold counts differ: {set(ext.n_folds)} vs {set(main.n_folds)}")


def edge_fractions(folds: pd.DataFrame) -> dict:
    c = folds.C_selected
    return {
        "frac_below_study_grid": float((c < min(STUDY)).mean()),
        "frac_above_study_grid": float((c > max(STUDY)).mean()),
        "frac_at_extended_min": float((c == min(EXTENDED)).mean()),
        "frac_at_extended_max": float((c == max(EXTENDED)).mean()),
    }


def _wide(scores: pd.DataFrame) -> pd.DataFrame:
    return scores.pivot(index="subject", columns="config", values="bacc")


def task_summary(task: str, main: pd.DataFrame, ext: pd.DataFrame, folds: pd.DataFrame) -> dict:
    a = _wide(main)
    b = _wide(ext).reindex_like(a)
    diff = b[BASELINE] - a[BASELINE]
    t = stats.ttest_1samp(diff, 0.0)
    return {
        "task": task,
        "n_subjects": len(a),
        "baseline_study": a[BASELINE].mean(),
        "baseline_extended": b[BASELINE].mean(),
        "baseline_diff": diff.mean(),
        "baseline_diff_t": t.statistic,
        "baseline_diff_p": t.pvalue,
        "all_configs_study": a.to_numpy().mean(),
        "all_configs_extended": b.to_numpy().mean(),
        "config_means_r": np.corrcoef(a.mean(), b.mean())[0, 1],
        **edge_fractions(folds),
    }


def planned_table(main: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    cols = ["contrast", "mean", "dz", "p_holm"]
    a = planned_contrasts(main, "lr")[cols].set_index("contrast")
    b = planned_contrasts(ext, "lr")[cols].set_index("contrast")
    out = a.add_suffix("_study").join(b.add_suffix("_extended"))
    out["significant_changed"] = (out.p_holm_study < 0.05) != (out.p_holm_extended < 0.05)
    return out.reset_index()


def joint_table(main: dict[str, pd.DataFrame], ext: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["contrast", "estimate", "dz", "p_holm", "heterogeneity_p_holm", "category"]
    a = joint_contrasts(main, "lr")[cols].set_index("contrast")
    b = joint_contrasts(ext, "lr")[cols].set_index("contrast")
    out = a.add_suffix("_study").join(b.add_suffix("_extended"))
    out["category_changed"] = out.category_study != out.category_extended
    return out.reset_index()


def main() -> None:
    out = results_root() / "c_grid_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    mains, exts, rows = {}, {}, []
    for task in TASKS:
        main_scores = load_main(task)
        ext_scores, folds = load_extended(task)
        check_complete(main_scores, ext_scores, task)
        mains[task], exts[task] = main_scores, ext_scores
        rows.append(task_summary(task, main_scores, ext_scores, folds))
        planned = planned_table(main_scores, ext_scores)
        planned.to_csv(out / f"planned_{task}.csv", index=False)
        print(f"== {task}: planned contrasts (LR), study vs extended grid")
        print(planned.round(4).to_string(index=False))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    joint = joint_table(mains, exts)
    joint.to_csv(out / "joint.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("== summary")
        print(summary.round(4).to_string(index=False))
        print("== joint test (LR), study vs extended grid")
        print(joint.round(4).to_string(index=False))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
