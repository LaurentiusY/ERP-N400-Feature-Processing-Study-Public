"""Models, cross-validation and per-subject decoding (sections 6.2-6.3).

* Outer folds: ``RepeatedStratifiedKFold(10, 5)`` with a fixed per-subject seed, generated
  once and saved, shared by every configuration and model of the subject.
* Logistic regression (lbfgs, L2): C chosen in each outer training fold by an inner
  stratified 5-fold CV on balanced accuracy; the normalization is part of the pipeline, so
  N2/N4 statistics are refitted inside the inner folds as well.
* Shrinkage LDA (lsqr, Ledoit-Wolf): no tuning.
* Stacking (guide 13.10, T5): :class:`StackedLR`, one logistic regression per feature block
  combined by a logistic-regression meta-model on their out-of-fold decision values.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DecodeConfig
from .normalize import make_normalizer
from .paths import subject_number

MODELS = ("lr", "lda")


def outer_splits(y: np.ndarray, subject: str, cfg: DecodeConfig = DecodeConfig()):
    """List of (train, test) index arrays: repeat r, fold f is element ``r * splits + f``."""
    rskf = RepeatedStratifiedKFold(
        n_splits=cfg.outer_splits,
        n_repeats=cfg.outer_repeats,
        random_state=cfg.seed + subject_number(subject),
    )
    return [(np.asarray(tr), np.asarray(te)) for tr, te in rskf.split(np.zeros(len(y)), y)]


def save_splits(path: Path, splits) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, (tr, te) in enumerate(splits):
        arrays[f"train_{k}"] = tr
        arrays[f"test_{k}"] = te
    np.savez(path, **arrays)


def load_splits(path: Path):
    with np.load(path) as f:
        n = len([k for k in f.files if k.startswith("train_")])
        return [(f[f"train_{k}"], f[f"test_{k}"]) for k in range(n)]


def get_splits(y: np.ndarray, subject: str, path: Path, cfg: DecodeConfig = DecodeConfig()):
    """Load the saved outer folds of the subject, or create and save them."""
    if path.exists():
        splits = load_splits(path)
        if sum(len(te) for _, te in splits) != len(y) * cfg.outer_repeats:
            raise RuntimeError(f"saved folds {path} do not match {len(y)} trials")
        return splits
    splits = outer_splits(y, subject, cfg)
    save_splits(path, splits)
    return splits


def make_estimator(model: str, norm: str, shape: tuple[int, int, int], cfg: DecodeConfig):
    if model == "lr":
        pipe = Pipeline(
            [
                ("norm", make_normalizer(norm, shape)),
                ("clf", LogisticRegression(solver="lbfgs", max_iter=cfg.lr_max_iter)),
            ]
        )
        return GridSearchCV(
            pipe,
            {"clf__C": list(cfg.c_grid)},
            cv=StratifiedKFold(cfg.inner_splits, shuffle=True, random_state=cfg.inner_seed),
            scoring="balanced_accuracy",
            refit=True,
            n_jobs=1,
        )
    if model == "lda":
        return Pipeline(
            [
                ("norm", make_normalizer(norm, shape)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    raise ValueError(f"unknown model {model!r}")


class StackedLR(ClassifierMixin, BaseEstimator):
    """Stacking of one logistic regression per feature block (guide 13.10, T5).

    ``x`` holds the blocks of ``parts`` (each ``(components, channels, bins)``) side by side.
    In ``fit`` (an outer training fold):

    1. each block's logistic regression is tuned as usual (normalization in the pipeline, C by
       the inner stratified 5-fold CV) and refitted on the whole training fold;
    2. with its C fixed, the same pipeline is cross-fitted (stratified 5-fold, fixed seed) to
       give every training trial an out-of-fold decision value;
    3. the meta-model, a logistic regression (C = 1) on the standardized out-of-fold decision
       values, is fitted.

    ``predict`` feeds the refitted block models' decision values to the meta-model. The C is
    fixed in step 2 so that the meta-model is trained and applied on decision values of the
    same model: re-tuning C inside each cross-fit fold gave decision values whose scale and
    offset differed several-fold from those of the refitted model, and the stack then
    predicted mostly one class. The standardization keeps the meta-model's penalty from
    shrinking the weight of a block tuned to a small C (tiny decision values) to zero.
    """

    def __init__(self, norm: str, parts, cfg: DecodeConfig = DecodeConfig()):
        self.norm = norm
        self.parts = parts
        self.cfg = cfg

    def _blocks(self, x: np.ndarray) -> list[np.ndarray]:
        edges = np.cumsum([0, *(int(np.prod(p)) for p in self.parts)])
        if x.shape[1] != edges[-1]:
            raise ValueError(f"expected {edges[-1]} features, got {x.shape[1]}")
        return [x[:, a:b] for a, b in zip(edges[:-1], edges[1:], strict=True)]

    def _inner_cv(self) -> StratifiedKFold:
        return StratifiedKFold(
            self.cfg.inner_splits, shuffle=True, random_state=self.cfg.inner_seed
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.models_, self.c_, oof = [], [], []
        for block, shape in zip(self._blocks(x), self.parts, strict=True):
            tuned = make_estimator("lr", self.norm, tuple(shape), self.cfg).fit(block, y)
            fixed = clone(tuned.best_estimator_)  # the pipeline with the selected C
            oof.append(
                cross_val_predict(fixed, block, y, cv=self._inner_cv(), method="decision_function")
            )
            self.models_.append(tuned.best_estimator_)
            self.c_.append(selected_c(tuned))
        self.meta_ = Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=self.cfg.lr_max_iter)),
            ]
        ).fit(np.column_stack(oof), y)
        return self

    def decision_values(self, x: np.ndarray) -> np.ndarray:
        """``(trials, blocks)`` decision values of the refitted block models."""
        return np.column_stack(
            [m.decision_function(b) for m, b in zip(self.models_, self._blocks(x), strict=True)]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.meta_.predict(self.decision_values(x))

    def details(self) -> dict:
        """Selected C of each block and the meta-model weights on the standardized decision
        values (``C_part<k>``, ``w_part<k>``)."""
        weights = self.meta_.named_steps["lr"].coef_[0]
        out = {}
        for k, (c, w) in enumerate(zip(self.c_, weights, strict=True)):
            out[f"C_part{k}"] = float(c)
            out[f"w_part{k}"] = float(w)
        return out


def make_stacking(norm: str, parts: list[tuple[int, int, int]], cfg: DecodeConfig) -> StackedLR:
    return StackedLR(norm, list(parts), cfg)


def selected_c(estimator) -> float:
    return float(estimator.best_params_["clf__C"]) if hasattr(estimator, "best_params_") else np.nan


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    splits,
    model: str,
    norm: str,
    shape: tuple[int, int, int],
    cfg: DecodeConfig = DecodeConfig(),
    parts: list[tuple[int, int, int]] | None = None,
) -> list[dict]:
    """One row per outer fold: repeat, fold, balanced accuracy, selected C, fit time.

    With ``parts`` (only for ``model="lr"``), ``x`` holds those blocks side by side and the
    model is the stacking of one logistic regression per block; the rows then also carry
    :meth:`StackedLR.details`, and ``C`` is the selected C of the first block.
    """
    if parts is not None and model != "lr":
        raise ValueError("stacking is defined for the logistic regression only")
    template = (
        make_estimator(model, norm, shape, cfg)
        if parts is None
        else make_stacking(norm, parts, cfg)
    )
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for k, (tr, te) in enumerate(splits):
            est = clone(template)
            t0 = time.perf_counter()
            est.fit(x[tr], y[tr])
            pred = est.predict(x[te])
            details = est.details() if parts is not None else {}
            rows.append(
                {
                    "repeat": k // cfg.outer_splits,
                    "fold": k % cfg.outer_splits,
                    "bacc": float(balanced_accuracy_score(y[te], pred)),
                    "C": details.get("C_part0", selected_c(est)),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "seconds": time.perf_counter() - t0,
                    **details,
                }
            )
    return rows
