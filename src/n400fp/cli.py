"""Command line: ``python -m n400fp <command> ...`` (or the ``n400fp`` script).

Commands (docs/README.md has the full workflow):

  download     ERP CORE N400 recordings
  preprocess   section 4, per subject
  bands        section 5, per subject
  tfr          Morlet coefficients for the time-frequency sensitivity analysis (13.10), per subject
  summary      preprocessing summary across subjects and the resolved channel sets
  decode       sections 6.1-6.3 for a stage (baseline | pilot | full | replication | tfr)
  permutation  per-subject permutation test of the baseline
  timecourse   time-resolved curves (section 6.5), per subject
  eegnet       EEGNet generality check, per subject
  stats        group statistics and reports of a stage
  joint        joint analysis of the three tasks (guide section 13.5), after ``stats --stage full``
  plan         list the configurations of a stage

Dataset: ``--dataset erpcore|scene|sentence`` before the command (default: ``N400_DATASET``, or
``erpcore``); e.g. ``python -m n400fp --dataset scene preprocess --subject sub-001``.
Subjects: ``--subject sub-001`` (repeatable), ``--subject-index 0`` (0-based, for SLURM
arrays), or ``--all`` (sub-001 ... sub-040).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import C_GRIDS, DecodeConfig
from .paths import DATASETS, N_SUBJECTS, dataset_name, deriv_dir, results_dir, subject_ids

log = logging.getLogger("n400fp")


def _subjects(args) -> list[str]:
    if getattr(args, "all", False):
        chosen = subject_ids(args.n_subjects)
    elif getattr(args, "subject_index", None) is not None:
        chosen = [subject_ids(args.n_subjects)[args.subject_index]]
    elif getattr(args, "subject", None):
        chosen = list(args.subject)
    else:
        raise SystemExit("give --subject, --subject-index or --all")
    if dataset_name() == "pursue":  # recordings without EEG data, excluded a priori (14.6)
        from . import pursue

        skipped = [s for s in chosen if pursue.excluded_a_priori(s)]
        for s in skipped:
            log.info("%s (%s): no EEG data, excluded a priori", s, pursue.subject_code(s))
        chosen = [s for s in chosen if s not in skipped]
    return chosen


def _add_subject_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--subject", action="append", help="e.g. sub-001 (repeatable)")
    p.add_argument("--subject-index", type=int, help="0-based index into sub-001, sub-002, ...")
    p.add_argument("--all", action="store_true", help="all subjects")
    p.add_argument("--n-subjects", type=int, help="default: all subjects of the dataset")
    p.add_argument("--n-jobs", type=int, default=1)


def cmd_download(args) -> None:
    from .download import download

    download(args.n_subjects)


def cmd_preprocess(args) -> None:
    from .preprocess import preproc_config, preprocess_subject

    cfg = preproc_config(dataset_name(), n_jobs=args.n_jobs)
    for s in _subjects(args):
        meta = preprocess_subject(s, cfg, overwrite=args.overwrite)
        if meta:
            log.info("%s: kept %d trials (%d/%d), ICA eog %d, muscle %d, interpolated %.3f, "
                     "rejected %d", s, meta["n_kept"], meta["n_related"], meta["n_unrelated"],
                     meta["ica_eog"]["n_components"], meta["ica_muscle"]["n_components"],
                     meta["interpolated_fraction"], meta["n_dropped_threshold"])  # fmt: skip


def cmd_bands(args) -> None:
    from .bands import compute_envelopes
    from .preprocess import preproc_config

    cfg = preproc_config(dataset_name(), n_jobs=args.n_jobs)
    for s in _subjects(args):
        compute_envelopes(s, cfg, overwrite=args.overwrite)


def cmd_tfr(args) -> None:
    from .preprocess import preproc_config
    from .tfr import compute_tfr

    cfg = preproc_config(dataset_name(), n_jobs=args.n_jobs)
    for s in _subjects(args):
        compute_tfr(s, cfg, overwrite=args.overwrite)


def cmd_summary(args) -> None:
    import pandas as pd

    from .summary import channel_sets, read_preproc

    chosen = args.subject or args.subject_index is not None
    table, channels = read_preproc(_subjects(args) if chosen else subject_ids())
    out = results_dir() / "reports" / "preprocessing"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "preproc_summary.csv", index=False)
    lists = {tuple(c) for c in channels.values()}
    if len(lists) > 1:
        log.warning("channel lists differ across subjects: %s", channels)
    sets = channel_sets(next(iter(channels.values())), allow_missing=dataset_name() == "pursue")
    (out / "channel_sets.json").write_text(json.dumps(sets, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
    print(f"\n{len(table)} subjects; channels ({len(next(iter(lists)))}):", *next(iter(lists)))
    for code, names in sets.items():
        print(f"{code} ({len(names)}):", *names)
    print(f"\nwritten to {out}")


def cmd_decode(args) -> None:
    from .run_subject import run_subject, subject_means
    from .study import Config

    configs = [Config.parse(c) for c in args.config] if args.config else None
    models = tuple(args.models.split(","))
    extra = {}
    if args.c_grid != "study":  # sensitivity analysis: LR only, separate folder, main folds
        models = ("lr",)
        extra = {
            "results_root": results_dir() / f"c_grid_{args.c_grid}",
            "splits_root": results_dir(),
            "dcfg": DecodeConfig(c_grid=C_GRIDS[args.c_grid]),
        }
        log.info("C grid %s %s -> %s", args.c_grid, C_GRIDS[args.c_grid], extra["results_root"])
    if args.stage == "tfr":  # sensitivity analysis (guide 13.10): LR only, own folder, main folds
        if args.c_grid != "study":
            raise SystemExit("the time-frequency analysis uses the study C grid")
        models = ("lr",)
        extra = {"results_root": results_dir() / "tfr_sensitivity", "splits_root": results_dir()}
    for s in _subjects(args):
        df = run_subject(s, args.stage, models=models, configs=configs, n_jobs=args.n_jobs, **extra)
        means = subject_means(df)
        for row in means.itertuples(index=False):
            if row.config == "B1-W1-N2-C1" or args.stage == "baseline":
                log.info("%s %s %s: bacc %.3f", s, row.model, row.config, row.bacc)


def cmd_permutation(args) -> None:
    import pandas as pd

    from .config import PermutationConfig
    from .decode import load_splits
    from .features import build_features, flatten, load_subject
    from .permutation import permutation_test
    from .run_subject import modal_c, run_subject, splits_path
    from .study import BASELINE, Config

    cfg = Config.parse(args.config or BASELINE)
    pcfg = PermutationConfig(n_permutations=args.n_permutations)
    out_dir = results_dir() / "permutation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in _subjects(args):
        data = load_subject(s, envelopes=cfg.B != "B1")
        main = run_subject(s, "baseline", models=("lr",), configs=[cfg], data=data)
        c_value = modal_c(main, cfg.code)
        folds = load_splits(splits_path(s))[:10]
        x4, _ = build_features(data, cfg)
        res = permutation_test(
            flatten(x4), data.y, folds, c_value, cfg.N, x4.shape[1:], pcfg, n_jobs=args.n_jobs
        )
        res.update({"subject": s, "config": cfg.code})
        pd.DataFrame([res]).to_parquet(out_dir / f"{s}.parquet", index=False)
        log.info("%s: observed %.3f, p %.4f (C=%g)", s, res["observed"], res["p"], c_value)


def cmd_timecourse(args) -> None:
    from .config import DecodeConfig
    from .decode import get_splits
    from .features import load_subject
    from .run_subject import splits_path
    from .timecourse import timecourse

    out_dir = results_dir() / "timecourse"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in _subjects(args):
        data = load_subject(s, envelopes=False)
        splits = get_splits(data.y, s, splits_path(s), DecodeConfig())
        timecourse(data, splits).to_parquet(out_dir / f"{s}.parquet", index=False)
        log.info("%s: time course saved", s)


def cmd_eegnet(args) -> None:
    from .eegnet import run_eegnet
    from .features import load_subject
    from .study import BASELINE, Config

    codes = list(args.config or [])
    if args.config_file:
        codes += [c.strip() for c in Path(args.config_file).read_text().split() if c.strip()]
    configs = [Config.parse(c) for c in (codes or [BASELINE])]
    out_dir = results_dir() / "eegnet"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in _subjects(args):
        data = load_subject(s, envelopes=any(c.B != "B1" for c in configs))
        run_eegnet(data, configs, device=args.device).to_parquet(out_dir / f"{s}.parquet")


def cmd_stats(args) -> None:
    from .collect import collect_folds, collect_table
    from .config import DecodeConfig
    from .run_subject import subject_means
    from .stats import (
        anova_table,
        art_anova,
        behaviour_table,
        config_tests,
        export_marginal_tables,
        gate_report,
        levels_for,
        pilot_report,
        planned_contrasts,
        timecourse_tests,
        to_array,
    )
    from .study import BASELINE, Config

    out = results_dir() / "reports" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    folds = collect_folds(args.stage)
    scores = subject_means(folds)
    scores.to_csv(out / "subject_scores.csv", index=False)
    config_tests(scores).to_csv(out / "config_tests.csv", index=False)

    if args.stage == "baseline":
        try:
            perm = collect_table("permutation")
            perm.to_csv(out / "permutation.csv", index=False)
        except FileNotFoundError:
            perm = None
        report = gate_report(scores, perm)
        (out / "gate.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return

    pilot_report(folds, DecodeConfig().c_grid).to_csv(out / "timing_and_C.csv", index=False)
    if args.stage == "pilot":
        return

    for model in sorted(scores.model.unique()):
        _, y = to_array(scores[scores.model == model], levels_for(model))
        anova_table(y).to_csv(out / f"rm_anova_{model}.csv", index=False)
        art_anova(y).to_csv(out / f"art_anova_{model}.csv", index=False)
        planned_contrasts(scores, model).to_csv(out / f"planned_contrasts_{model}.csv", index=False)
        export_marginal_tables(scores, model, out / "marginal_tables")

    if dataset_name() == "erpcore":  # the Draschkow tasks have no semantic judgement responses
        subjects = sorted(scores.subject.unique())
        per_subject, quantiles = behaviour_table(deriv_dir(), subjects)
        per_subject.to_csv(out / "behaviour_subjects.csv", index=False)
        quantiles.to_csv(out / "behaviour_rt_quantiles.csv", index=False)
    try:
        timecourse_tests(collect_table("timecourse")).to_csv(out / "timecourse.csv", index=False)
    except FileNotFoundError:
        log.warning("no time-course results yet")

    lr = scores[scores.model == "lr"].groupby("config").bacc.mean().sort_values(ascending=False)
    series = [c for c in lr.index if Config.parse(c).B in ("B1", "B2", "B3", "B4", "B5")]
    chosen = list(dict.fromkeys([BASELINE] + series[:10] + series[-3:]))
    (out / "eegnet_configs.txt").write_text("\n".join(chosen) + "\n")


def cmd_joint(args) -> None:
    import pandas as pd

    from .joint import (
        TASKS,
        config_agreement,
        effect_size_table,
        joint_contrasts,
        shared_subject_correlation,
    )
    from .paths import results_root, task_results_dir
    from .study import BASELINE

    reports = {t: task_results_dir(t) / "reports" / "full" for t in TASKS}
    missing = [str(r) for r in reports.values() if not (r / "subject_scores.csv").exists()]
    if missing:
        raise SystemExit(f"run `stats --stage full` for every dataset first; missing: {missing}")
    scores = {t: pd.read_csv(r / "subject_scores.csv") for t, r in reports.items()}
    tests = {t: pd.read_csv(r / "config_tests.csv") for t, r in reports.items()}
    out = results_root() / "joint"
    out.mkdir(parents=True, exist_ok=True)
    for model in ("lr", "lda"):
        table = joint_contrasts(scores, model)
        table.to_csv(out / f"joint_contrasts_{model}.csv", index=False)
        anova = {t: pd.read_csv(r / f"rm_anova_{model}.csv") for t, r in reports.items()}
        effect_size_table(anova).to_csv(out / f"effect_sizes_{model}.csv", index=False)
        agreement = config_agreement(tests, model)
        agreement["correlations"].to_csv(out / f"config_correlations_{model}.csv", index=False)
        agreement["ranking"].to_csv(out / f"config_ranking_{model}.csv", index=False)
        cols = ["contrast", "estimate", "ci_low", "ci_high", "p_holm", "heterogeneity_p_holm"]
        print(f"== {model}")
        print(table[[*cols, "category"]].round(4).to_string(index=False))
    base = {}
    for t in ("scene", "sentence"):
        s = scores[t]
        base[t] = s[(s.model == "lr") & (s.config == BASELINE)].set_index("subject").bacc
    shared = shared_subject_correlation(base["scene"], base["sentence"])
    (out / "scene_sentence_subject_correlation.json").write_text(json.dumps(shared, indent=2))
    print(f"written to {out}")


def cmd_plan(args) -> None:
    from .study import model_configs, stage_configs

    configs = stage_configs(args.stage)
    for model in ("lr", "lda"):
        print(f"{model}: {len(model_configs(configs, model))} configurations")
    if args.verbose:
        print("\n".join(c.code for c in configs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="n400fp", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--dataset", choices=DATASETS, help="default: N400_DATASET, or erpcore (guide section 13)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download")
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    p.set_defaults(func=cmd_download)

    for name, func in (("preprocess", cmd_preprocess), ("bands", cmd_bands), ("tfr", cmd_tfr)):
        p = sub.add_parser(name)
        _add_subject_args(p)
        p.add_argument("--overwrite", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("summary", help="preprocessing summary (default: every subject found)")
    _add_subject_args(p)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("decode")
    _add_subject_args(p)
    p.add_argument(
        "--stage",
        choices=["baseline", "pilot", "full", "replication", "tfr"],
        required=True,
        help="'replication': the three preregistered PURSUE configurations (guide 14.4); "
        "'tfr': the time-frequency sensitivity analysis, LR only, <results>/tfr_sensitivity "
        "(guide 13.10)",
    )
    p.add_argument("--models", default="lr,lda")
    p.add_argument("--config", action="append", help="restrict to these codes (repeatable)")
    p.add_argument(
        "--c-grid",
        choices=list(C_GRIDS),
        default="study",
        help="LR grid; 'extended' is the sensitivity analysis (LR only, <results>/c_grid_extended)",
    )
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("permutation")
    _add_subject_args(p)
    p.add_argument("--config", help="default: the baseline")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.set_defaults(func=cmd_permutation)

    p = sub.add_parser("timecourse")
    _add_subject_args(p)
    p.set_defaults(func=cmd_timecourse)

    p = sub.add_parser("eegnet")
    _add_subject_args(p)
    p.add_argument("--config", action="append")
    p.add_argument("--config-file", help="one code per line, e.g. reports/full/eegnet_configs.txt")
    p.add_argument("--device", default="cpu")
    p.set_defaults(func=cmd_eegnet)

    p = sub.add_parser("stats")
    p.add_argument("--stage", choices=["baseline", "pilot", "full"], required=True)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("joint", help="joint analysis of the three tasks (after stats --stage full)")
    p.set_defaults(func=cmd_joint)

    p = sub.add_parser("plan")
    p.add_argument("--stage", choices=["baseline", "pilot", "full", "replication", "tfr"],
                   required=True)  # fmt: skip
    p.set_defaults(func=cmd_plan)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dataset:
        os.environ["N400_DATASET"] = args.dataset
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    from .paths import preproc_variant

    if preproc_variant() != "main":  # validated here, before any command reads or writes
        log.warning("N400_PREPROC=%s: preprocessing variant of guide 13.11", preproc_variant())
    args.func(args)


if __name__ == "__main__":
    main()
