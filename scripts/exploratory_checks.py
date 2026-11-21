"""Post hoc exploratory analyses of the full study (not part of the pre-specified inference).

Reads the full-stage reports, the per-subject epochs in ``$N400_DERIV_DIR`` and the EEGNet
results, and writes ``<results>/reports/full/exploratory/``:

1. ``sensitivity_planned_lr.csv``: planned contrasts without the low-quality subjects
   (ERP CORE: sub-029, 80 trials left, and sub-040, excluded by ERP CORE for artifacts;
   sentence: sub-017, 33 trials left after the threshold rejection; scene: none, skipped);
2. ``condition_effects.csv``: per subject, unrelated - related in the C2 region: waveform
   (uV and single-trial Cohen's d) and log10 band envelopes (baseline-corrected), in
   300-500 and 500-800 ms (envelopes only where ``env-*-epo.fif`` exist);
   ``condition_effects_group.csv``: one-sample t tests;
3. ``correlations.csv``: subject-level correlations (Pearson and Spearman) between baseline
   accuracy, ERP effects, envelope accuracy, EEGNet accuracy and (ERP CORE only) the RT effect;
4. ``representation_correlations.csv``: subject-level correlations between the LR marginal
   means of the eight representations;
5. ``lda_minus_lr.csv``: LDA - LR per shared configuration with its number of features.

    python scripts/exploratory_checks.py              # N400_DATASET selects the dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from n400fp.collect import collect_table
from n400fp.config import BAND_ORDER
from n400fp.features import load_subject, log_baseline, resolve
from n400fp.paths import dataset_name, deriv_dir, results_dir, subject_ids
from n400fp.stats import planned_contrasts
from n400fp.study import BASELINE, CHANNEL_SETS, REPRESENTATIONS, Config

N_CHANNELS = {"C1": 28, "C2": 7, "C3": 7, "C4": 7, "C5": 17}
N_BINS = {"W1": 12, "W2": 4, "W3": 6, "W4": 16}
WINDOWS = {"n400": (0.3, 0.5), "late": (0.5, 0.8)}
LOW_QUALITY = {"erpcore": ["sub-029", "sub-040"], "scene": [], "sentence": ["sub-017"]}


def condition_effects() -> pd.DataFrame:
    rows = []
    for s in subject_ids():
        envelopes = (deriv_dir() / s / "env-theta-epo.fif").exists()
        d = load_subject(s, envelopes=envelopes)
        idx = resolve(list(CHANNEL_SETS["C2"]), d.ch_names)
        rel, unrel = d.y == 0, d.y == 1
        wave = d.wave[:, idx].mean(axis=1)
        logs = (
            {b: log_baseline(d.env[b], d.env_times)[:, idx].mean(axis=1) for b in BAND_ORDER}
            if envelopes
            else {}
        )
        row = {"subject": s}
        for name, (lo, hi) in WINDOWS.items():
            amp = wave[:, (d.wave_times >= lo) & (d.wave_times < hi)].mean(axis=1)
            diff = amp[unrel].mean() - amp[rel].mean()
            sd = np.sqrt((amp[unrel].var(ddof=1) + amp[rel].var(ddof=1)) / 2)
            row[f"wave_{name}_uV"] = diff
            row[f"wave_{name}_d"] = diff / sd
            for band, lg in logs.items():
                v = lg[:, (d.env_times >= lo) & (d.env_times < hi)].mean(axis=1)
                row[f"{band}_{name}_log10"] = v[unrel].mean() - v[rel].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("subject")


def one_sample(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in effects.columns:
        v = effects[col]
        t = stats.ttest_1samp(v, 0.0)
        rows.append(
            {
                "measure": col,
                "mean": v.mean(),
                "sd": v.std(ddof=1),
                "t": t.statistic,
                "p": t.pvalue,
                "n_negative": int((v < 0).sum()),
                "n": len(v),
            }
        )
    return pd.DataFrame(rows)


def correlation(x: pd.Series, y: pd.Series, label: str) -> dict:
    x, y = x.align(y, join="inner")
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return {"pair": label, "r": r, "p": p, "rho": rho, "p_rho": p_rho, "n": len(x)}


def main() -> None:
    reports = results_dir() / "reports" / "full"
    out = reports / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(reports / "subject_scores.csv")

    def acc(model: str, code: str) -> pd.Series:
        g = scores[(scores.model == model) & (scores.config == code)]
        return g.set_index("subject").bacc

    # 1. sensitivity
    excluded = LOW_QUALITY[dataset_name()]
    sens = pd.DataFrame()
    if excluded:
        a = planned_contrasts(scores, "lr").set_index("contrast")
        kept = scores[~scores.subject.isin(excluded)]
        b = planned_contrasts(kept, "lr").set_index("contrast")
        n_all, n_kept = scores.subject.nunique(), kept.subject.nunique()
        sens = pd.DataFrame(
            {
                f"mean_{n_all}": a["mean"],
                f"dz_{n_all}": a.dz,
                f"p_holm_{n_all}": a.p_holm,
                f"mean_{n_kept}": b["mean"],
                f"dz_{n_kept}": b.dz,
                f"p_holm_{n_kept}": b.p_holm,
            }
        )
        sens.to_csv(out / "sensitivity_planned_lr.csv")

    # 2. condition effects
    eff = condition_effects()
    eff.to_csv(out / "condition_effects.csv")
    one_sample(eff).to_csv(out / "condition_effects_group.csv", index=False)

    # 3. correlations
    base = acc("lr", BASELINE)
    marg = pd.read_csv(reports / "marginal_tables" / "lr_B.csv")
    by_b = marg.pivot(index="subject", columns="B", values="acc")
    by_b.corr().to_csv(out / "representation_correlations.csv")
    eeg = collect_table("eegnet")
    eeg_base = eeg[eeg.config == BASELINE].groupby("subject").bacc.mean()
    late = acc("lr", "B1-W3-N2-C1")
    rows = [
        correlation(eff.wave_n400_uV, base, "N400 effect (uV) ~ baseline LR accuracy"),
        correlation(eff.wave_n400_d, base, "N400 effect (d) ~ baseline LR accuracy"),
        correlation(eff.wave_late_d, base, "late effect (d) ~ baseline LR accuracy"),
        correlation(
            by_b.B1,
            by_b[["B2", "B3", "B4", "B5"]].mean(axis=1),
            "B1 ~ mean single-band envelope accuracy (LR marginal means)",
        ),
        correlation(eeg_base, base, "EEGNet ~ LR accuracy at the baseline configuration"),
    ]
    if dataset_name() == "erpcore":  # the Draschkow tasks have no responses
        beh = pd.read_csv(reports / "behaviour_subjects.csv").set_index("subject")
        rt_effect = beh.median_rt_unrelated - beh.median_rt_related
        rows += [
            correlation(rt_effect, base, "RT effect ~ baseline LR accuracy"),
            correlation(rt_effect, late, "RT effect ~ late-window LR accuracy (B1-W3-N2-C1)"),
            correlation(eff.wave_n400_d, rt_effect, "N400 effect (d) ~ RT effect"),
        ]
    pd.DataFrame(rows).to_csv(out / "correlations.csv", index=False)
    diff = (base - eeg_base).dropna()
    t = stats.ttest_1samp(diff, 0.0)
    pd.DataFrame(
        [
            {
                "mean_lr_minus_eegnet": diff.mean(),
                "t": t.statistic,
                "p": t.pvalue,
                "n_lr_better": int((diff > 0).sum()),
                "n": len(diff),
            }
        ]
    ).to_csv(out / "eegnet_vs_lr_baseline.csv", index=False)

    # 5. LDA - LR by number of features
    ct = pd.read_csv(reports / "config_tests.csv")
    lr = ct[ct.model == "lr"].set_index("config").mean_bacc
    lda = ct[ct.model == "lda"].set_index("config").mean_bacc
    pair = pd.DataFrame({"lda": lda, "lr": lr.reindex(lda.index)}).dropna()
    cfgs = [Config.parse(c) for c in pair.index]
    pair["n_features"] = [len(REPRESENTATIONS[c.B]) * N_CHANNELS[c.C] * N_BINS[c.W] for c in cfgs]
    pair["waveform"] = [c.B in ("B1", "B8") for c in cfgs]
    pair["lda_minus_lr"] = pair.lda - pair.lr
    pair.to_csv(out / "lda_minus_lr.csv")

    print(sens.round(4).to_string())
    print(one_sample(eff).round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    w = pair[pair.waveform]
    print(w.groupby(pd.cut(w.n_features, [0, 50, 150, 500, 3000])).lda_minus_lr.mean().round(4))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
