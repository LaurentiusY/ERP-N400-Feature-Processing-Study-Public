"""Merge per-subject results of a stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import results_dir


def collect_folds(stage: str, root: Path | None = None) -> pd.DataFrame:
    """All per-fold rows of ``<results>/<stage>/sub-*/*.parquet``."""
    base = (root or results_dir()) / stage
    files = sorted(base.glob("sub-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no results under {base}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def collect_table(name: str, root: Path | None = None) -> pd.DataFrame:
    """Concatenate ``<results>/<name>/sub-*.parquet`` (permutation, timecourse, eegnet)."""
    base = (root or results_dir()) / name
    files = sorted(base.glob("sub-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no results under {base}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
