"""D3 normalizations as scikit-learn transformers (section 6.1).

Inputs are flattened ``(trials, components * channels * bins)`` arrays in C order, with the
4-D shape ``(components, channels, bins)`` passed to the transformer. Statistics that span
trials (N2, N4) are fitted on the training trials only; inside a Pipeline that includes the
inner cross-validation of the logistic regression.

* N1 - no normalization.
* N2 - per-feature standardization (``StandardScaler``), statistics from the training fold.
* N3 - per-trial z-score within each component block (the trial's own mean and SD).
* N4 - multivariate noise normalization (Guggenmos et al., 2018): per component, subtract
  the class means of the training trials, estimate the channel x channel noise covariance
  over all bins with Ledoit-Wolf shrinkage, and left-multiply every bin's channel vector by
  its inverse square root.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _as4d(x: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.shape[1] != int(np.prod(shape)):
        raise ValueError(f"{x.shape[1]} features do not match shape {shape}")
    return x.reshape(len(x), *shape)


class PerTrialZ(TransformerMixin, BaseEstimator):
    """N3: z-score each trial within each component block."""

    def __init__(self, shape: tuple[int, int, int]):
        self.shape = shape

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x4 = _as4d(x, self.shape)
        flat = x4.reshape(len(x4), self.shape[0], -1)
        mean = flat.mean(axis=-1, keepdims=True)
        sd = flat.std(axis=-1, keepdims=True)
        sd = np.where(sd > 0, sd, 1.0)
        return ((flat - mean) / sd).reshape(len(x4), -1)


def inverse_sqrt(cov: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, np.finfo(float).eps * vals.max())
    return (vecs / np.sqrt(vals)) @ vecs.T


class MVNN(TransformerMixin, BaseEstimator):
    """N4: spatial noise whitening per component block."""

    def __init__(self, shape: tuple[int, int, int]):
        self.shape = shape

    def fit(self, x, y):
        if y is None:
            raise ValueError("MVNN needs class labels to estimate the noise")
        x4 = _as4d(x, self.shape)
        y = np.asarray(y)
        self.whiteners_ = []
        for b in range(self.shape[0]):
            xb = x4[:, b]  # (trials, channels, bins)
            resid = np.empty_like(xb)
            for label in np.unique(y):
                sel = y == label
                resid[sel] = xb[sel] - xb[sel].mean(axis=0, keepdims=True)
            noise = resid.transpose(0, 2, 1).reshape(-1, self.shape[1])
            cov = LedoitWolf(assume_centered=True).fit(noise).covariance_
            self.whiteners_.append(inverse_sqrt(cov))
        self.whiteners_ = np.stack(self.whiteners_)
        return self

    def transform(self, x):
        x4 = _as4d(x, self.shape)
        return np.einsum("bij,nbjt->nbit", self.whiteners_, x4).reshape(len(x4), -1)


def make_normalizer(code: str, shape: tuple[int, int, int]):
    if code == "N1":
        return FunctionTransformer(None)  # identity
    if code == "N2":
        return StandardScaler()
    if code == "N3":
        return PerTrialZ(shape)
    if code == "N4":
        return MVNN(shape)
    raise ValueError(f"unknown normalization {code!r}")
