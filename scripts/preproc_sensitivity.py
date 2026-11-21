"""Sensitivity analysis: do the results depend on the preprocessing changes made after gate 1?

The full stage of the two Draschkow tasks is rerun with the preprocessing specified before gate 1
(guide section 13.11; ``N400_PREPROC=prespecified``, results in
``<results>/preproc_prespecified/<task>/``). This script compares it with the main run (guide
section 13.8), for both models; ERP CORE is unchanged and enters both joint tests as is:

1. ``summary.csv``: per task and model, the baseline and all-configuration mean accuracy under
   both pipelines and the correlation of the configuration means;
2. ``planned_<task>.csv``: the 11 planned contrasts of each task under both pipelines;
3. ``joint.csv``: the joint three-task test under both pipelines, and whether a category changed.

The main analysis stays the one of guide section 13.8; this is a robustness check. Output in
``<results>/preproc_sensitivity/``.

    python scripts/preproc_sensitivity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from n400fp.joint import TASKS, joint_contrasts
from n400fp.paths import VARIANT_TASKS, prespecified_results_dir, results_root, task_results_dir
from n400fp.stats import planned_contrasts
from n400fp.study import BASELINE, Config, full_factorial, model_configs

MODELS = ("lr", "lda")


def baseline_code(model: str) -> str:
    """The baseline as the model runs it (LDA: N1, the equivalent of N2)."""
    return model_configs([Config.parse(BASELINE)], model)[0].code


def load_scores(root) -> pd.DataFrame:
    return pd.read_csv(root / "reports" / "full" / "subject_scores.csv")


def check_complete(main: pd.DataFrame, alt: pd.DataFrame, task: str) -> None:
    """Same subjects in both runs and, per model, every configuration for every subject."""
    missing, extra = set(main.subject) - set(alt.subject), set(alt.subject) - set(main.subject)
    if missing:
        raise SystemExit(f"{task}: pre-specified preprocessing missing subjects {sorted(missing)}")
    if extra:
        raise SystemExit(f"{task}: subjects only in the pre-specified run {sorted(extra)}")
    for model in MODELS:
        expected = {c.code for c in model_configs(full_factorial(), model)}
        for label, scores in (("main", main), ("pre-specified", alt)):
            found = scores[scores.model == model].groupby("subject").config.agg(set)
            short = sorted(s for s, configs in found.items() if configs != expected)
            if short or len(found) != main.subject.nunique():
                detail = short or "missing model"
                raise SystemExit(f"{task}/{model}: incomplete {label} subjects {detail}")


def with_prespecified(
    mains: dict[str, pd.DataFrame], prespecified: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """The three tasks of the second joint test: ERP CORE's main scores, the Draschkow reruns."""
    if set(prespecified) != set(VARIANT_TASKS):
        raise ValueError(f"expected reruns of {VARIANT_TASKS}, got {sorted(prespecified)}")
    return {task: prespecified.get(task, mains[task]) for task in TASKS}


def _wide(scores: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = scores[scores.model == model]
    return rows.pivot(index="subject", columns="config", values="bacc")


def task_summary(task: str, main: pd.DataFrame, alt: pd.DataFrame, model: str) -> dict:
    a = _wide(main, model)
    b = _wide(alt, model).reindex_like(a)
    base = baseline_code(model)
    diff = b[base] - a[base]
    t = stats.ttest_1samp(diff, 0.0)
    return {
        "task": task,
        "model": model,
        "n_subjects": len(a),
        "baseline_config": base,
        "baseline_main": a[base].mean(),
        "baseline_prespecified": b[base].mean(),
        "baseline_diff": diff.mean(),
        "baseline_diff_t": t.statistic,
        "baseline_diff_p": t.pvalue,
        "all_configs_main": a.to_numpy().mean(),
        "all_configs_prespecified": b.to_numpy().mean(),
        "config_means_r": np.corrcoef(a.mean(), b.mean())[0, 1],
    }


def planned_table(main: pd.DataFrame, alt: pd.DataFrame, model: str) -> pd.DataFrame:
    cols = ["contrast", "mean", "dz", "p_holm"]
    a = planned_contrasts(main, model)[cols].set_index("contrast")
    b = planned_contrasts(alt, model)[cols].set_index("contrast")
    out = a.add_suffix("_main").join(b.add_suffix("_prespecified"))
    out["significant_changed"] = (out.p_holm_main < 0.05) != (out.p_holm_prespecified < 0.05)
    out["sign_changed"] = np.sign(out.mean_main) != np.sign(out.mean_prespecified)
    out.insert(0, "model", model)
    return out.reset_index()


def joint_table(
    main: dict[str, pd.DataFrame], alt: dict[str, pd.DataFrame], model: str
) -> pd.DataFrame:
    cols = ["contrast", "estimate", "dz", "p_holm", "heterogeneity_p_holm", "category"]
    a = joint_contrasts(main, model)[cols].set_index("contrast")
    b = joint_contrasts(alt, model)[cols].set_index("contrast")
    out = a.add_suffix("_main").join(b.add_suffix("_prespecified"))
    out["category_changed"] = out.category_main != out.category_prespecified
    out.insert(0, "model", model)
    return out.reset_index()


def main() -> None:
    out = results_root() / "preproc_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    mains = {task: load_scores(task_results_dir(task)) for task in TASKS}
    alts = with_prespecified(
        mains, {task: load_scores(prespecified_results_dir(task)) for task in VARIANT_TASKS}
    )
    for task in VARIANT_TASKS:
        check_complete(mains[task], alts[task], task)

    summary = pd.DataFrame(
        [task_summary(t, mains[t], alts[t], m) for t in VARIANT_TASKS for m in MODELS]
    )
    summary.to_csv(out / "summary.csv", index=False)
    for task in VARIANT_TASKS:
        planned = pd.concat([planned_table(mains[task], alts[task], m) for m in MODELS])
        planned.to_csv(out / f"planned_{task}.csv", index=False)
    joint = pd.concat([joint_table(mains, alts, m) for m in MODELS], ignore_index=True)
    joint.to_csv(out / "joint.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("== summary")
        print(summary.round(4).to_string(index=False))
        print("== joint test, main vs pre-specified preprocessing")
        print(joint.round(4).to_string(index=False))
    changed = joint[joint.category_changed]
    if changed.empty:
        print("RESULT: no joint category changed (LR, LDA); the conclusions do not depend on the "
              "preprocessing changes after gate 1")  # fmt: skip
    else:
        print("RESULT: joint categories changed:")
        print(changed[["model", "contrast", "category_main", "category_prespecified"]]
              .to_string(index=False))  # fmt: skip
    print(f"written to {out}")


if __name__ == "__main__":
    main()
