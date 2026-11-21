"""Fixed settings of the study (docs/engineering_guide.md sections 3-6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- recording (ERP CORE N400, BIDS channels.tsv) -------------------------------------------
EEG_CHANNELS = [
    "FP1", "F3", "F7", "FC3", "C3", "C5", "P3", "P7", "P9", "PO7", "PO3", "O1", "Oz", "Pz", "CPz",
    "FP2", "Fz", "F4", "F8", "FC4", "FCz", "Cz", "C4", "C6", "P4", "P8", "P10", "PO8", "PO4", "O2",
]  # fmt: skip
EOG_CHANNELS = ["HEOG_left", "HEOG_right", "VEOG_lower"]
REFERENCE = ["P9", "P10"]
ERPCORE_ELECTRODES = tuple(c for c in EEG_CHANNELS if c not in REFERENCE)  # 28 recorded

# --- events ---------------------------------------------------------------------------------
RELATED = ("211", "212")
UNRELATED = ("221", "222")
PRIMES = ("111", "112", "121", "122")
RESPONSE_CORRECT = "201"
RESPONSE_INCORRECT = "202"
RESPONSES = (RESPONSE_CORRECT, RESPONSE_INCORRECT)
LCD_DELAY = 0.026  # s, added to every stimulus event (Kessler et al.; ERP CORE scripts)
LABELS = {"related": 0, "unrelated": 1}

# --- frequency bands of the envelopes (section 5) --------------------------------------------
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0), "gamma": (30.0, 45.0)}
BAND_ORDER = ["theta", "alpha", "beta", "gamma"]

# --- time-frequency sensitivity analysis (section 13.10) -------------------------------------
# complex Morlet wavelets on the continuous data: 12 log-spaced frequencies, 3-10 cycles
TFR_FREQS = tuple(float(f) for f in np.geomspace(2.0, 36.0, 12))
TFR_CYCLES = tuple(float(n) for n in np.geomspace(3.0, 10.0, 12))
TFR_LOW_MAX = 8.0  # Hz: T2 and T3 use the frequencies up to here (2.00-7.44 Hz)


@dataclass(frozen=True)
class PreprocConfig:
    """Kessler's Luck path for N400: ``ica_ica_None_0.1_P9P10_None_200ms_int``."""

    sfreq: float = 256.0
    initial_reference: tuple[str, ...] = ("Cz",)
    ica_n_components: int = 20
    ica_method: str = "picard"
    ica_random_state: int = 97
    ica_highpass: float = 1.0
    highpass: float = 0.1
    reference: tuple[str, ...] = tuple(REFERENCE)
    epoch_tmin: float = -0.4
    epoch_tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    autoreject_subsample: float = 0.25
    autoreject_subsample_seed: int = 12
    autoreject_seed: int = 11
    autoreject_cv: int = 5
    autoreject_min_kept: float = 0.2
    # Draschkow tasks (guide section 13.8): muscle ICA, bad-trial handling ("interpolate" =
    # autoreject as above, "reject" = drop trials above a global peak-to-peak threshold) and
    # the channels kept in the saved epochs (empty = all; matched case-insensitively)
    muscle_ica: bool = True
    bad_trials: str = "interpolate"
    flat_sd: float = 1e-6  # V; "reject": trials whose median channel SD is lower are dropped first
    keep_channels: tuple[str, ...] = ()
    # envelopes (section 5)
    env_tmin: float = -0.5  # envelope epochs end at the waveform epochs' last sample
    n_jobs: int = 1


@dataclass(frozen=True)
class FeatureConfig:
    bin_width: float = 0.05
    env_baseline: tuple[float, float] = (-0.5, -0.2)
    eps: float = 1e-12  # floor before log10 (interpolated envelopes can dip to ~0)


C_GRID_STUDY = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
# logistic-regression grids: the pre-specified one and the sensitivity analysis (both ends
# extended; results of the latter go to <results>/c_grid_extended/, never the main results)
C_GRIDS = {"study": C_GRID_STUDY, "extended": (1e-5, 1e-4, *C_GRID_STUDY, 1e3, 1e4)}


@dataclass(frozen=True)
class DecodeConfig:
    outer_splits: int = 10
    outer_repeats: int = 5
    inner_splits: int = 5
    c_grid: tuple[float, ...] = C_GRID_STUDY
    lr_max_iter: int = 5000
    seed: int = 20261011  # outer folds use seed + subject number
    inner_seed: int = 7


@dataclass(frozen=True)
class PermutationConfig:
    n_permutations: int = 1000
    seed: int = 4242


@dataclass(frozen=True)
class EEGNetConfig:
    n_splits: int = 5
    seeds: tuple[int, ...] = (0, 1, 2)
    max_epochs: int = 200
    batch_size: int = 16
