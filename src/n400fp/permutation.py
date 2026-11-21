"""Per-subject label-permutation test of the baseline configuration (stages 2 and 4).

The group test (section 6.4 step 1) is the gate; this test adds, for each subject, whether
its own decoding is above chance. To keep the cost low, observed and permuted scores are
computed the same way: the first repeat of the subject's outer folds (10 folds), logistic
regression with a fixed C (the C selected most often for this subject in the main run) and
the configuration's normalization. ``p = (1 + #{null >= observed}) / (1 + n_permutations)``.
"""

from __future__ import annotations

import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

from .config import DecodeConfig, PermutationConfig
from .normalize import make_normalizer


def cv_score(x, y, folds, estimator) -> float:
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        for tr, te in folds:
            est = clone(estimator).fit(x[tr], y[tr])
            scores.append(balanced_accuracy_score(y[te], est.predict(x[te])))
    return float(np.mean(scores))


def _null_scores(x, y, folds, estimator, seeds) -> list[float]:
    out = []
    for s in seeds:
        perm = np.random.default_rng(s).permutation(y)
        out.append(cv_score(x, perm, folds, estimator))
    return out


def permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    folds,
    c_value: float,
    norm: str,
    shape: tuple[int, int, int],
    pcfg: PermutationConfig = PermutationConfig(),
    dcfg: DecodeConfig = DecodeConfig(),
    n_jobs: int = 1,
) -> dict:
    estimator = Pipeline(
        [
            ("norm", make_normalizer(norm, shape)),
            ("clf", LogisticRegression(C=c_value, solver="lbfgs", max_iter=dcfg.lr_max_iter)),
        ]
    )
    observed = cv_score(x, y, folds, estimator)
    seeds = np.random.SeedSequence(pcfg.seed).generate_state(pcfg.n_permutations)
    chunks = np.array_split(seeds, max(1, n_jobs))
    null = np.concatenate(
        Parallel(n_jobs=n_jobs)(
            delayed(_null_scores)(x, y, folds, estimator, chunk) for chunk in chunks
        )
    )
    p = (1 + np.sum(null >= observed)) / (1 + len(null))
    return {
        "observed": observed,
        "p": float(p),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_95": float(np.quantile(null, 0.95)),
        "C_fixed": float(c_value),
        "n_permutations": int(len(null)),
    }
