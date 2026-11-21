"""Preregistered PURSUE replication (engineering guide section 14, tag ``prereg-pursue-v1``).

Reads the decoding results of the three registered configurations (``decode --stage
replication``, ``results/pursue/replication/``)
and the preprocessing records (``derivatives/pursue/sub-XXX/preproc.json``) and tests, on the
per-subject mean balanced accuracy over the 50 outer folds:

* R1: ``B8-W1-N2-C1`` < ``B1-W1-N2-C1`` (adding band envelopes lowers accuracy; P1);
* R2: ``B1-W1-N2-C1`` > ``B2-W1-N2-C1`` (waveform better than the theta envelope; P3);

one-sided paired t tests, Holm over R1 and R2, alpha 0.05, Wilcoxon signed-rank tests as a
check; the replication succeeds if both are significant in the predicted direction (logistic
regression, sample after the exclusions of 14.6). Also reported: the baseline against chance
(manipulation check), the results without the post-processing exclusion, per site, and for
shrinkage LDA (N1 in place of the equivalent N2). Output in
``results/pursue/reports/replication/``.

    python scripts/pursue_replication.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from n400fp import pursue
from n400fp.collect import collect_folds
from n400fp.paths import deriv_dir, results_dir
from n400fp.run_subject import subject_means
from n400fp.stats import holm

ALPHA = 0.05
N_FOLDS = 50
NORM = {"lr": "N2", "lda": "N1"}  # LDA is invariant to N2, its identical N1 is run instead
HYPOTHESES = (  # name, (minuend, subtrahend) representations, predicted sign, question
    ("R1", ("B8", "B1"), -1, "B8 - B1 < 0: band envelopes added to the waveform lower accuracy"),
    ("R2", ("B1", "B2"), +1, "B1 - B2 > 0: waveform better than the theta envelope"),
)


def config(b: str, model: str) -> str:
    return f"{b}-W1-{NORM[model]}-C1"


def one_sided(d: pd.Series, sign: int) -> dict:
    """Paired test of ``d`` against 0 in the direction ``sign``; mean, 95% CI, dz, t, p."""
    n = len(d)
    t = stats.ttest_1samp(d, 0.0, alternative="less" if sign < 0 else "greater")
    w = stats.wilcoxon(d, alternative="less" if sign < 0 else "greater") if n > 1 else None
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return {
        "n": n,
        "mean": d.mean(),
        "ci_low": d.mean() - half,
        "ci_high": d.mean() + half,
        "dz": d.mean() / d.std(ddof=1),
        "t": t.statistic,
        "p": t.pvalue,
        "p_wilcoxon": w.pvalue if w is not None else np.nan,
    }


def replication_tests(scores: pd.DataFrame, model: str, subjects: list[str]) -> pd.DataFrame:
    """R1, R2 (Holm) and the baseline-against-chance check for one model and sample."""
    wide = scores[scores.model == model].pivot(index="subject", columns="config", values="bacc")
    wide = wide.loc[subjects]
    rows = []
    for name, (a, b), sign, question in HYPOTHESES:
        d = wide[config(a, model)] - wide[config(b, model)]
        rows.append(
            {"test": name, "question": question, "predicted_sign": sign, **one_sided(d, sign)}
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["p_wilcoxon_holm"] = holm(out.p_wilcoxon.to_numpy())
    out["replicated"] = (out.p_holm < ALPHA) & (np.sign(out["mean"]) == out.predicted_sign)
    base = wide[config("B1", model)] - 0.5
    check = {"test": "baseline > 0.5", "question": "manipulation check", "predicted_sign": 1}
    out = pd.concat([out, pd.DataFrame([{**check, **one_sided(base, 1)}])], ignore_index=True)
    out.insert(0, "model", model)
    return out


def subject_table(root: Path) -> pd.DataFrame:
    """Per preprocessed subject: original label, site, trials per class, post-processing
    inclusion (at least 30 trials in each class)."""
    rows = []
    for path in sorted(root.glob("sub-*/preproc.json")):
        meta = json.loads(path.read_text())
        code = meta["subject_code"]
        rows.append(
            {
                "subject": meta["subject"],
                "code": code,
                "site": pursue.site(code),
                "n_related": meta["n_related"],
                "n_unrelated": meta["n_unrelated"],
                "included": min(meta["n_related"], meta["n_unrelated"])
                >= pursue.MIN_TRIALS_PER_CLASS,
            }
        )
    return pd.DataFrame(rows)


def check_complete(scores: pd.DataFrame, subjects: list[str]) -> None:
    for model in NORM:
        need = {config(b, model) for b in ("B1", "B2", "B8")}
        got = scores[scores.model == model]
        for s in subjects:
            have = got[got.subject == s]
            if set(have.config) < need or (have.n_folds != N_FOLDS).any():
                raise SystemExit(f"{s}: incomplete {model} results {sorted(have.config)}")


def main() -> None:
    os.environ["N400_DATASET"] = "pursue"
    out = results_dir() / "reports" / "replication"
    out.mkdir(parents=True, exist_ok=True)
    subjects = subject_table(deriv_dir())
    scores = subject_means(collect_folds("replication"))
    decoded = sorted(set(scores.subject) & set(subjects.subject))
    check_complete(scores, decoded)
    subjects = subjects[subjects.subject.isin(decoded)]
    subjects.to_csv(out / "subjects.csv", index=False)

    samples = {
        "main (14.6 exclusions)": subjects[subjects.included].subject.tolist(),
        "without post-processing exclusion": subjects.subject.tolist(),
    }
    tests = pd.concat(
        [
            replication_tests(scores, model, ids).assign(sample=name)
            for name, ids in samples.items()
            for model in NORM
        ],
        ignore_index=True,
    )
    tests.to_csv(out / "tests.csv", index=False)
    by_site = pd.concat(
        [
            replication_tests(scores, "lr", g[g.included].subject.tolist()).assign(site=site)
            for site, g in subjects.groupby("site")
        ],
        ignore_index=True,
    )
    by_site.to_csv(out / "by_site.csv", index=False)

    counts = subjects.groupby("site").agg(decoded=("subject", "size"), included=("included", "sum"))
    cols = ["sample", "model", "test", "n", "mean", "ci_low", "ci_high", "dz", "t", "p", "p_holm",
            "p_wilcoxon", "replicated"]  # fmt: skip
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(counts.to_string())
        print(tests[cols].round(4).to_string(index=False))
        print(
            by_site[["site", "test", "n", "mean", "dz", "p", "p_holm"]]
            .round(4)
            .to_string(index=False)
        )
    main_lr = tests[(tests["sample"] == "main (14.6 exclusions)") & (tests.model == "lr")]
    verdict = main_lr[main_lr.test.isin(["R1", "R2"])].replicated.all()
    result = "SUCCEEDED" if verdict else "NOT replicated"
    print(f"\nreplication (LR, main sample, R1 and R2): {result}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
