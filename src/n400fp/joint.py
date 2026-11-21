"""Joint analysis of the three semantic tasks (engineering guide section 13.5).

The confirmatory inference tests each planned contrast P1-P11 as the equally weighted mean of
its effect in the three tasks, theta = (theta_erpcore + theta_scene + theta_sentence) / 3.
ERP CORE participants are independent of the Draschkow participants, while ``scene`` and
``sentence`` share theirs; each Draschkow participant therefore enters through the average of
their two task differences, a = (d_scene + d_sentence) / 2:

    theta_hat = mean(d_erpcore) / 3 + 2 * mean(a) / 3
    SE^2      = var(d_erpcore) / (9 n_erpcore) + 4 var(a) / (9 n_draschkow)

with Welch-Satterthwaite degrees of freedom, a two-sided t test and Holm over the 11
contrasts. Heterogeneity (equal effects in the three tasks) is a 2-df Wald test on the
contrasts theta_scene - theta_sentence (paired) and theta_erpcore - (theta_scene +
theta_sentence) / 2 (independent groups), with Holm over the 11 contrasts.

Each contrast is classified as
* ``general``: joint effect significant (Holm), same sign in all three tasks, no heterogeneity;
* ``same direction, size varies``: joint effect significant, same sign, heterogeneity;
* ``task-dependent``: signs differ and heterogeneity is significant;
* ``no reliable general effect``: none of the above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import PLANNED, contrast_differences, holm, summarize

TASKS = ("erpcore", "scene", "sentence")
CATEGORIES = (
    "general",
    "same direction, size varies",
    "task-dependent",
    "no reliable general effect",
)


def joint_test(d_erpcore: pd.Series, d_scene: pd.Series, d_sentence: pd.Series) -> dict:
    """Equally weighted joint effect of one contrast; ``d_*`` are per-subject differences."""
    shared = d_scene.index.intersection(d_sentence.index)
    ds, dt = d_scene.loc[shared], d_sentence.loc[shared]
    a = (ds + dt) / 2
    de = d_erpcore.dropna()
    n_e, n_f = len(de), len(a)

    estimate = de.mean() / 3 + 2 * a.mean() / 3
    v_e = de.var(ddof=1) / (9 * n_e)
    v_f = 4 * a.var(ddof=1) / (9 * n_f)
    se = np.sqrt(v_e + v_f)
    df = (v_e + v_f) ** 2 / (v_e**2 / (n_e - 1) + v_f**2 / (n_f - 1))
    t = estimate / se
    half = stats.t.ppf(0.975, df) * se
    sd = np.sqrt((de.var(ddof=1) + d_scene.var(ddof=1) + d_sentence.var(ddof=1)) / 3)

    # heterogeneity: c1 = scene - sentence (paired), c2 = erpcore - mean(scene, sentence)
    diff = ds - dt
    c = np.array([diff.mean(), de.mean() - a.mean()])
    cov12 = -np.cov(diff, a, ddof=1)[0, 1] / n_f
    v = np.array(
        [
            [diff.var(ddof=1) / n_f, cov12],
            [cov12, de.var(ddof=1) / n_e + a.var(ddof=1) / n_f],
        ]
    )
    wald = float(c @ np.linalg.solve(v, c))
    return {
        "estimate": estimate,
        "ci_low": estimate - half,
        "ci_high": estimate + half,
        "dz": estimate / sd if sd > 0 else np.nan,
        "se": se,
        "df": df,
        "t": t,
        "p": 2 * stats.t.sf(abs(t), df),
        "heterogeneity_wald": wald,
        "heterogeneity_p": stats.chi2.sf(wald, 2),
        "n_erpcore": n_e,
        "n_draschkow": n_f,
    }


def classify(p_holm: float, signs: list[float], het_p_holm: float) -> str:
    same = all(s > 0 for s in signs) or all(s < 0 for s in signs)
    if p_holm < 0.05 and same:
        return CATEGORIES[0] if het_p_holm >= 0.05 else CATEGORIES[1]
    if not same and het_p_holm < 0.05:
        return CATEGORIES[2]
    return CATEGORIES[3]


def joint_contrasts(scores: dict[str, pd.DataFrame], model: str) -> pd.DataFrame:
    """Joint test of P1-P11 for one model; ``scores`` maps each task to its subject scores."""
    per_task = {}
    for task in TASKS:
        subjects, diffs, _ = contrast_differences(scores[task], model)
        per_task[task] = {k: pd.Series(v, index=subjects) for k, v in diffs.items()}
    rows = []
    for code, label, question in PLANNED:
        row = {"model": model, "contrast": code, "definition": label, "question": question}
        row.update(joint_test(*(per_task[task][code] for task in TASKS)))
        for task in TASKS:
            s = summarize(per_task[task][code].to_numpy())
            row[f"{task}_mean"] = s["mean"]
            row[f"{task}_ci_low"] = s["ci_low"]
            row[f"{task}_ci_high"] = s["ci_high"]
            row[f"{task}_dz"] = s["dz"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["heterogeneity_p_holm"] = holm(out.heterogeneity_p.to_numpy())
    out["category"] = [
        classify(r.p_holm, [r.erpcore_mean, r.scene_mean, r.sentence_mean], r.heterogeneity_p_holm)
        for r in out.itertuples()
    ]
    return out


def effect_size_table(anova: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generalized eta squared of every effect side by side, with the mean over the tasks."""
    table = pd.DataFrame({task: anova[task].set_index("effect").ges for task in TASKS})
    table["mean"] = table[list(TASKS)].mean(axis=1)
    return table.reset_index()


def config_agreement(config_tests: dict[str, pd.DataFrame], model: str) -> dict[str, pd.DataFrame]:
    """Correlations of configuration accuracies between tasks; ranking on the task average."""
    acc = pd.DataFrame(
        {t: c[c.model == model].set_index("config").mean_bacc for t, c in config_tests.items()}
    )[list(TASKS)].dropna()
    corr = pd.DataFrame(
        [
            {
                "task_a": a,
                "task_b": b,
                "pearson": acc[a].corr(acc[b]),
                "spearman": acc[a].corr(acc[b], method="spearman"),
            }
            for i, a in enumerate(TASKS)
            for b in TASKS[i + 1 :]
        ]
    )
    ranking = acc.assign(mean=acc.mean(axis=1)).sort_values("mean", ascending=False)
    return {"correlations": corr, "ranking": ranking.reset_index()}


def shared_subject_correlation(scene: pd.Series, sentence: pd.Series) -> dict:
    """Same participants: correlation of their accuracies in the two Draschkow tasks."""
    x, y = scene.align(sentence, join="inner")
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"pearson": r, "p": p, "spearman": rho, "p_spearman": p_rho, "n": len(x)}
