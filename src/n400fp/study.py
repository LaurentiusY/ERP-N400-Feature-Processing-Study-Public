"""Factors, configuration codes and the configuration sets of each stage (section 6.1).

A configuration code looks like ``B1-W1-N2-C1`` (the baseline).
"""

from __future__ import annotations

import itertools
from typing import NamedTuple

from .config import TFR_FREQS, TFR_LOW_MAX

# D1: components of each representation. "wave" = waveform (uV); a band name = that band's
# log envelope, baseline-corrected; "rel:<band>" = log relative amplitude, baseline-corrected.
REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "B1": ("wave",),
    "B2": ("theta",),
    "B3": ("alpha",),
    "B4": ("beta",),
    "B5": ("gamma",),
    "B6": ("theta", "alpha", "beta", "gamma"),
    "B7": ("rel:theta", "rel:alpha", "rel:beta", "rel:gamma"),
    "B8": ("wave", "theta", "alpha", "beta", "gamma"),
}

# Time-frequency sensitivity analysis (guide 13.10), not part of the factorial design.
# "amp:k" = log amplitude of wavelet frequency k, baseline-corrected; "cos:k" / "sin:k" =
# cosine / sine of its phase. T5 stacks the logistic regressions of B1 and T1.
_TFR_ALL = tuple(range(len(TFR_FREQS)))
_TFR_LOW = tuple(k for k, f in enumerate(TFR_FREQS) if f <= TFR_LOW_MAX)
TFR_REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "T1": tuple(f"amp:{k}" for k in _TFR_ALL),
    "T2": tuple(f"amp:{k}" for k in _TFR_LOW),
    "T3": tuple(c for k in _TFR_LOW for c in (f"cos:{k}", f"sin:{k}")),
    "T4": ("wave", *(f"amp:{k}" for k in _TFR_ALL)),
}
STACKED: dict[str, tuple[str, str]] = {"T5": ("B1", "T1")}
TFR_LEVELS = (*TFR_REPRESENTATIONS, *STACKED)
TFR_FIXED = ("W1", "N2", "C1")

# D2: windows in seconds (bins of 50 ms)
WINDOWS: dict[str, tuple[float, float]] = {
    "W1": (0.2, 0.8),
    "W2": (0.3, 0.5),
    "W3": (0.5, 0.8),
    "W4": (0.0, 0.8),
}

# D3: normalizations (implemented in n400fp.normalize)
NORMALIZATIONS = ("N1", "N2", "N3", "N4")

# D4: channel sets; names are matched case-insensitively against the data.
REFERENCE_CHANNELS = ("P9", "P10")
CHANNEL_SETS: dict[str, tuple[str, ...] | None] = {
    "C1": None,  # every scalp channel except the reference
    "C2": ("Cz", "CPz", "Pz", "C3", "C4", "P3", "P4"),
    "C3": ("F7", "F3", "Fz", "F4", "F8", "FC3", "FC4"),
    "C4": ("PO3", "PO4", "PO7", "PO8", "O1", "O2", "Oz"),
    "C5": None,  # left-right symmetrized montage, see SYMMETRIC_PAIRS / MIDLINE
}
SYMMETRIC_PAIRS = (
    ("FP1", "FP2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC3", "FC4"),
    ("C3", "C4"),
    ("C5", "C6"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("PO3", "PO4"),
    ("PO7", "PO8"),
    ("O1", "O2"),
)
MIDLINE = ("Fz", "FCz", "Cz", "CPz", "Pz", "Oz")

FACTORS = {
    "B": tuple(REPRESENTATIONS),
    "W": tuple(WINDOWS),
    "N": NORMALIZATIONS,
    "C": tuple(CHANNEL_SETS),
}
BASELINE = "B1-W1-N2-C1"
REPLICATION = (BASELINE, "B8-W1-N2-C1", "B2-W1-N2-C1")  # PURSUE (guide 14.4)
LDA_NORMALIZATIONS = ("N1", "N3", "N4")  # LDA is invariant to N2 (identical to N1)


class Config(NamedTuple):
    B: str
    W: str
    N: str
    C: str

    @property
    def code(self) -> str:
        return f"{self.B}-{self.W}-{self.N}-{self.C}"

    @classmethod
    def parse(cls, code: str) -> Config:
        parts = code.strip().split("-")
        if len(parts) != 4:
            raise ValueError(f"bad configuration code {code!r}")
        cfg = cls(*parts)
        for name, value in zip("BWNC", cfg, strict=True):
            allowed = FACTORS[name] + (TFR_LEVELS if name == "B" else ())
            if value not in allowed:
                raise ValueError(f"unknown level {value!r} in {code!r}")
        return cfg

    @property
    def feature_key(self) -> tuple[str, str, str]:
        """Configurations sharing (B, W, C) share their features before normalization."""
        return (self.B, self.W, self.C)


def full_factorial() -> list[Config]:
    return [Config(*levels) for levels in itertools.product(*FACTORS.values())]


def pilot() -> list[Config]:
    """The baseline plus every configuration that changes exactly one factor (18)."""
    base = Config.parse(BASELINE)
    out = [base]
    for i, name in enumerate("BWNC"):
        for level in FACTORS[name]:
            if level != base[i]:
                out.append(base._replace(**{name: level}))
    return out


def stage_configs(stage: str) -> list[Config]:
    if stage == "baseline":
        return [Config.parse(BASELINE)]
    if stage == "pilot":
        return pilot()
    if stage == "full":
        return full_factorial()
    if stage == "replication":  # PURSUE, preregistered (guide 14.4)
        return [Config.parse(c) for c in REPLICATION]
    if stage == "tfr":  # time-frequency sensitivity analysis (guide 13.10), LR only
        return [Config(b, *TFR_FIXED) for b in TFR_LEVELS]
    raise ValueError(f"unknown stage {stage!r}")


def components(b: str) -> tuple[str, ...]:
    """All feature components a representation needs (a stacked one: those of its parts)."""
    if b in STACKED:
        return tuple(c for part in STACKED[b] for c in components(part))
    return REPRESENTATIONS[b] if b in REPRESENTATIONS else TFR_REPRESENTATIONS[b]


def is_tfr_component(component: str) -> bool:
    return component.split(":")[0] in ("amp", "cos", "sin")


def model_configs(configs: list[Config], model: str) -> list[Config]:
    """Configurations a model runs: LDA replaces N2 by its equivalent N1 (deduplicated)."""
    if model != "lda":
        return list(configs)
    seen, out = set(), []
    for cfg in configs:
        cfg = cfg._replace(N="N1") if cfg.N == "N2" else cfg
        if cfg not in seen:
            seen.add(cfg)
            out.append(cfg)
    return out
