"""ERP CORE N400 download and raw file lookup."""

from __future__ import annotations

from pathlib import Path

from .paths import N_SUBJECTS, data_dir


def download(n_subjects: int = N_SUBJECTS, root: Path | None = None):
    """Download the N400 recordings with hu-neuro-pipeline (as Kessler et al. did).

    Files land in ``<root>/erpcore/N400/sub-XXX/eeg/*.set``. If the package cannot be
    installed, any copy of the EEGLAB files under ``root`` works too (for example the BIDS
    version, ``sub-XXX/eeg/sub-XXX_task-N400_eeg.set``): :func:`find_raw_file` searches
    recursively.
    """
    try:
        from pipeline.datasets import get_erpcore
    except ImportError as err:  # pragma: no cover - optional dependency
        raise ImportError(
            "hu-neuro-pipeline is needed for the download: pip install -e .[download]. "
            "Alternatively place the ERP CORE N400 .set/.fdt files under N400_DATA_DIR."
        ) from err
    root = Path(root or data_dir())
    root.mkdir(parents=True, exist_ok=True)
    return get_erpcore("N400", participants=n_subjects, path=str(root))


def find_raw_file(subject: str, root: Path | None = None) -> Path:
    """The single N400 ``.set`` file of ``subject`` below ``root``."""
    root = Path(root or data_dir())
    hits = sorted(
        p for p in root.rglob("*.set") if subject in p.as_posix() and "n400" in p.as_posix().lower()
    )
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one N400 .set file for {subject} under {root}, found {len(hits)}: {hits}"
        )
    return hits[0]
