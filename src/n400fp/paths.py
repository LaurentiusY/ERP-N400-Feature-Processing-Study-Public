"""Data, derivative and result folders (environment variables, with repository defaults).

The dataset is chosen with ``N400_DATASET`` (``erpcore``, ``scene``, ``sentence`` or
``pursue``; default ``erpcore``; the command line sets it with ``--dataset``). ERP CORE keeps the
top-level ``derivatives/`` and ``results/`` folders; the other datasets use subfolders named
after the dataset, e.g. ``derivatives/scene/sub-001`` and ``results/pursue/full``.

``N400_PREPROC=prespecified`` (Draschkow tasks only; guide section 13.11) selects the
preprocessing specified before gate 1 for the sensitivity analysis; its derivatives and results
go to ``<derivatives|results>/preproc_prespecified/<task>/`` and leave the main ones untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

N_SUBJECTS = 40  # ERP CORE and the Draschkow tasks
DATASETS = ("erpcore", "scene", "sentence", "pursue")
SUBJECTS_PER_DATASET = {"erpcore": 40, "scene": 40, "sentence": 40, "pursue": 288}
PREPROC_VARIANTS = ("main", "prespecified")
VARIANT_TASKS = ("scene", "sentence")  # the tasks whose preprocessing changed after gate 1
PRESPECIFIED_DIR = "preproc_prespecified"


def _dir(env: str, default: str | Path) -> Path:
    return Path(os.environ.get(env, REPO_ROOT / default)).expanduser()


def dataset_name() -> str:
    name = os.environ.get("N400_DATASET", "erpcore")
    if name not in DATASETS:
        raise ValueError(f"N400_DATASET must be one of {DATASETS}, not {name!r}")
    return name


def preproc_variant() -> str:
    """``N400_PREPROC``: ``main`` (default) or ``prespecified`` (scene and sentence only)."""
    variant = os.environ.get("N400_PREPROC") or "main"
    if variant not in PREPROC_VARIANTS:
        raise ValueError(f"N400_PREPROC must be one of {PREPROC_VARIANTS}, not {variant!r}")
    if variant != "main" and dataset_name() not in VARIANT_TASKS:
        raise ValueError(f"N400_PREPROC={variant} applies to {VARIANT_TASKS}, not {dataset_name()}")
    return variant


def _scoped(base: Path) -> Path:
    name = dataset_name()
    if preproc_variant() == "prespecified":
        return base / PRESPECIFIED_DIR / name
    return base if name == "erpcore" else base / name


def data_dir() -> Path:
    """Raw ERP CORE downloads (``N400_DATA_DIR``, default ``<repo>/data``)."""
    return _dir("N400_DATA_DIR", "data")


def pursue_dir() -> Path:
    """The PURSUE N400 BIDS dataset (NEMAR on007052), the folder holding ``sub-1001/`` etc.
    (``N400_PURSUE_DIR``, default ``<data>/external/pursue``)."""
    default = data_dir() / "external" / "pursue"
    return Path(os.environ.get("N400_PURSUE_DIR", default)).expanduser()


def draschkow_dir() -> Path:
    """Extracted Draschkow et al. (2018) data, the folder holding ``unprocessed/``
    (``N400_DRASCHKOW_DIR``, default ``<data>/external/draschkow2018/extracted/n3n4``)."""
    default = data_dir() / "external" / "draschkow2018" / "extracted" / "n3n4"
    return Path(os.environ.get("N400_DRASCHKOW_DIR", default)).expanduser()


def deriv_dir() -> Path:
    """Preprocessed data of the current dataset (``N400_DERIV_DIR``, default
    ``<repo>/derivatives``; a subfolder per Draschkow task)."""
    return _scoped(_dir("N400_DERIV_DIR", "derivatives"))


def results_root() -> Path:
    """Top results folder (``N400_RESULTS_DIR``, default ``<repo>/results``)."""
    return _dir("N400_RESULTS_DIR", "results")


def results_dir() -> Path:
    """Decoding results and statistics of the current dataset (a subfolder of
    :func:`results_root` per Draschkow task)."""
    return _scoped(results_root())


def task_results_dir(dataset: str) -> Path:
    """Results folder of a given dataset, whatever ``N400_DATASET`` is."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    return results_root() if dataset == "erpcore" else results_root() / dataset


def prespecified_results_dir(task: str) -> Path:
    """Results of a Draschkow task under the pre-gate-1 preprocessing (guide section 13.11)."""
    if task not in VARIANT_TASKS:
        raise ValueError(f"no pre-specified preprocessing variant for {task!r}")
    return results_root() / PRESPECIFIED_DIR / task


def n_subjects(dataset: str | None = None) -> int:
    """Number of participants of a dataset (default: the current one)."""
    return SUBJECTS_PER_DATASET[dataset or dataset_name()]


def subject_ids(n: int | None = None) -> list[str]:
    """``sub-001`` ... for the current dataset (or the first ``n``)."""
    return [f"sub-{i:03d}" for i in range(1, (n or n_subjects()) + 1)]


def subject_deriv(subject: str, root: Path | None = None) -> Path:
    path = (root or deriv_dir()) / subject
    path.mkdir(parents=True, exist_ok=True)
    return path


def subject_number(subject: str) -> int:
    """``"sub-007"`` -> 7."""
    return int(subject.split("-")[-1])
