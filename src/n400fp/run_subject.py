"""Decode every configuration of one subject for one stage (one SLURM array task).

Results go to ``<results>/<stage>/<subject>/<B>-<W>-<C>.parquet``, one file per feature group
(all normalizations and models that share B, W and C). Existing complete files are skipped,
so a job that timed out can simply be resubmitted. Outer folds are stored once per subject
in ``<results>/splits/<subject>.npz`` and reused by every stage; a sensitivity run with its
own ``results_root`` reads them from ``splits_root`` (the main results) and never creates them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import DecodeConfig, FeatureConfig
from .decode import MODELS, evaluate, get_splits
from .features import SubjectData, build_features, flatten, load_subject
from .paths import results_dir
from .study import STACKED, Config, components, is_tfr_component, model_configs, stage_configs

log = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "subject", "model", "config", "B", "W", "N", "C",
    "repeat", "fold", "bacc", "C_selected", "n_train", "n_test", "n_features", "seconds",
]  # fmt: skip
# stacking only (T5, guide 13.10): selected C and meta-model weight of each part
STACKING_COLUMNS = ["C_part0", "C_part1", "w_part0", "w_part1"]


def splits_path(subject: str, root: Path | None = None) -> Path:
    return (root or results_dir()) / "splits" / f"{subject}.npz"


def group_path(stage: str, subject: str, key: tuple[str, str, str], root: Path | None) -> Path:
    return (root or results_dir()) / stage / subject / f"{'-'.join(key)}.parquet"


def plan(configs: list[Config], models: tuple[str, ...]) -> dict[tuple, list[tuple[str, Config]]]:
    """Feature group (B, W, C) -> list of (model, configuration) to evaluate."""
    groups: dict[tuple, list[tuple[str, Config]]] = defaultdict(list)
    for model in models:
        for cfg in model_configs(configs, model):
            groups[cfg.feature_key].append((model, cfg))
    return dict(groups)


def run_group(
    data: SubjectData,
    key: tuple[str, str, str],
    jobs: list[tuple[str, Config]],
    splits,
    dcfg: DecodeConfig,
    fcfg: FeatureConfig,
) -> pd.DataFrame:
    any_cfg = jobs[0][1]
    parts = None
    if any_cfg.B in STACKED:  # the blocks of the stacked representations, side by side
        blocks = [build_features(data, any_cfg._replace(B=b), fcfg)[0] for b in STACKED[any_cfg.B]]
        parts = [tuple(int(s) for s in b.shape[1:]) for b in blocks]
        shape = parts[0]
        x = np.hstack([flatten(b) for b in blocks])
    else:
        x4, _ = build_features(data, any_cfg, fcfg)
        shape = tuple(int(s) for s in x4.shape[1:])
        x = flatten(x4)
    rows = []
    for model, cfg in jobs:
        for r in evaluate(x, data.y, splits, model, cfg.N, shape, dcfg, parts=parts):
            rows.append(
                {
                    "subject": data.subject,
                    "model": model,
                    "config": cfg.code,
                    "B": cfg.B,
                    "W": cfg.W,
                    "N": cfg.N,
                    "C": cfg.C,
                    "repeat": r["repeat"],
                    "fold": r["fold"],
                    "bacc": r["bacc"],
                    "C_selected": r["C"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                    "n_features": int(x.shape[1]),
                    "seconds": r["seconds"],
                    **{k: r[k] for k in STACKING_COLUMNS if k in r},
                }
            )
    extra = [k for k in STACKING_COLUMNS if parts is not None]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS + extra)


def _run_and_save(data, key, jobs, splits, dcfg, fcfg, path: Path) -> None:
    df = run_group(data, key, jobs, splits, dcfg, fcfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _complete(path: Path, jobs: list[tuple[str, Config]], n_folds: int) -> bool:
    if not path.exists():
        return False
    df = pd.read_parquet(path, columns=["model", "config"])
    expected = {(m, c.code) for m, c in jobs}
    counts = df.groupby(["model", "config"]).size()
    return set(counts.index) >= expected and all(counts[k] == n_folds for k in expected)


def run_subject(
    subject: str,
    stage: str,
    models: tuple[str, ...] = MODELS,
    configs: list[Config] | None = None,
    n_jobs: int = 1,
    deriv_root: Path | None = None,
    results_root: Path | None = None,
    dcfg: DecodeConfig = DecodeConfig(),
    fcfg: FeatureConfig = FeatureConfig(),
    data: SubjectData | None = None,
    splits_root: Path | None = None,
) -> pd.DataFrame:
    """Run (or resume) one subject; returns all rows of the stage for this subject.

    ``splits_root``: read the outer folds from there (they must exist) instead of
    ``results_root``, so that a sensitivity run uses exactly the folds of the main run.
    """
    configs = configs if configs is not None else stage_configs(stage)
    comps = {c for cfg in configs for c in components(cfg.B)}
    needs_tfr = any(is_tfr_component(c) for c in comps)
    needs_env = any(c != "wave" and not is_tfr_component(c) for c in comps)
    data = data or load_subject(subject, deriv_root, envelopes=needs_env, tfr=needs_tfr)
    if splits_root is not None and not splits_path(subject, splits_root).exists():
        raise FileNotFoundError(f"no saved folds for {subject} in {splits_root}")
    splits = get_splits(data.y, subject, splits_path(subject, splits_root or results_root), dcfg)
    n_folds = len(splits)
    groups = plan(configs, models)
    todo = {
        k: v
        for k, v in groups.items()
        if not _complete(group_path(stage, subject, k, results_root), v, n_folds)
    }
    log.info("%s/%s: %d feature groups, %d to run", subject, stage, len(groups), len(todo))

    Parallel(n_jobs=n_jobs)(
        delayed(_run_and_save)(
            data, k, v, splits, dcfg, fcfg, group_path(stage, subject, k, results_root)
        )
        for k, v in todo.items()
    )
    parts = [pd.read_parquet(group_path(stage, subject, k, results_root)) for k in groups]
    out = pd.concat(parts, ignore_index=True)
    wanted = {(m, c.code) for v in groups.values() for m, c in v}
    out = out[[(m, c) in wanted for m, c in zip(out.model, out.config, strict=True)]]
    return out.reset_index(drop=True)


def subject_means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over outer folds: one row per subject x model x configuration."""
    keys = ["subject", "model", "config", "B", "W", "N", "C"]
    agg = df.groupby(keys, as_index=False).agg(
        bacc=("bacc", "mean"), n_folds=("bacc", "size"), seconds=("seconds", "mean")
    )
    return agg


def modal_c(df: pd.DataFrame, config: str) -> float:
    """Most frequently selected C of the logistic regression for ``config`` (ties: smaller)."""
    sel = df[(df.model == "lr") & (df.config == config)].C_selected.dropna()
    if sel.empty:
        return 1.0
    counts = sel.value_counts()
    best = counts[counts == counts.max()].index
    return float(np.min(best))
