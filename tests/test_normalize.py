import numpy as np
import pytest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from n400fp.features import build_features, flatten
from n400fp.normalize import MVNN, PerTrialZ, make_normalizer
from n400fp.study import Config


@pytest.fixture(scope="module")
def baseline_xy(subject):
    x4, _ = build_features(subject, Config("B1", "W1", "N1", "C1"))
    return flatten(x4), subject.y, tuple(x4.shape[1:])


def _split(n):
    idx = np.arange(n)
    return idx[: int(0.8 * n)], idx[int(0.8 * n) :]


def test_lda_invariant_to_per_feature_standardization(baseline_xy):
    x, y, shape = baseline_xy
    tr, te = _split(len(y))
    out = {}
    for norm in ("N1", "N2", "N3"):
        pipe = Pipeline(
            [
                ("norm", make_normalizer(norm, shape)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ).fit(x[tr], y[tr])
        out[norm] = pipe.decision_function(x[te])
    assert np.allclose(out["N1"], out["N2"], rtol=1e-6, atol=1e-8)
    assert not np.allclose(out["N1"], out["N3"])


def test_lr_differs_across_normalizations(baseline_xy):
    x, y, shape = baseline_xy
    tr, te = _split(len(y))
    out = {}
    for norm in ("N1", "N2", "N3", "N4"):
        pipe = Pipeline(
            [
                ("norm", make_normalizer(norm, shape)),
                ("clf", LogisticRegression(C=1.0, max_iter=5000)),
            ]
        ).fit(x[tr], y[tr])
        out[norm] = pipe.decision_function(x[te])
    norms = list(out)
    for i, a in enumerate(norms):
        for b in norms[i + 1 :]:
            assert not np.allclose(out[a], out[b]), (a, b)


def test_n3_is_computed_per_block(subject):
    x4, _ = build_features(subject, Config("B8", "W2", "N1", "C2"))
    shape = tuple(x4.shape[1:])
    z = PerTrialZ(shape).fit_transform(flatten(x4)).reshape(x4.shape)
    per_block = z.reshape(len(z), shape[0], -1)
    assert np.allclose(per_block.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(per_block.std(axis=-1), 1.0)
    # single block: identical to the whole-trial z-score
    x1, _ = build_features(subject, Config("B1", "W2", "N1", "C2"))
    flat = flatten(x1)
    z1 = PerTrialZ(tuple(x1.shape[1:])).fit_transform(flat)
    manual = (flat - flat.mean(1, keepdims=True)) / flat.std(1, keepdims=True)
    assert np.allclose(z1, manual)


def _noise_cov(x4b, y):
    resid = np.empty_like(x4b)
    for label in (0, 1):
        resid[y == label] = x4b[y == label] - x4b[y == label].mean(axis=0)
    noise = resid.transpose(0, 2, 1).reshape(-1, x4b.shape[1])
    return noise.T @ noise / len(noise)


def _mean_abs_corr(cov):
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return np.abs(corr[~np.eye(len(cov), dtype=bool)]).mean()


def test_mvnn_whitens_training_noise(baseline_xy):
    from sklearn.covariance import LedoitWolf

    x, y, shape = baseline_xy
    mvnn = MVNN(shape).fit(x, y)
    w = mvnn.whiteners_[0]
    # exact: the whitener maps the shrunk noise covariance to the identity
    before = x.reshape(len(x), *shape)[:, 0]
    resid = np.empty_like(before)
    for label in (0, 1):
        resid[y == label] = before[y == label] - before[y == label].mean(axis=0)
    noise = resid.transpose(0, 2, 1).reshape(-1, shape[1])
    cov_lw = LedoitWolf(assume_centered=True).fit(noise).covariance_
    assert np.allclose(w @ cov_lw @ w.T, np.eye(shape[1]), atol=1e-8)
    # approximately: the empirical noise correlations largely disappear
    after = mvnn.transform(x).reshape(len(x), *shape)[:, 0]
    assert _mean_abs_corr(_noise_cov(after, y)) < 0.5 * _mean_abs_corr(_noise_cov(before, y))


def test_fitted_statistics_ignore_test_trials(baseline_xy):
    x, y, shape = baseline_xy
    tr, te = _split(len(y))
    x2 = x.copy()
    x2[te] = x2[te] * 10 + 3  # change only the test trials
    for norm in ("N2", "N4"):
        a = make_normalizer(norm, shape).fit(x[tr], y[tr])
        b = make_normalizer(norm, shape).fit(x2[tr], y[tr])
        assert np.allclose(a.transform(x[te]), b.transform(x[te]))
    # N3 never looks at other trials
    n3 = make_normalizer("N3", shape)
    assert np.allclose(n3.fit_transform(x)[tr], n3.fit_transform(x2)[tr])


def test_normalizers_clone():
    for norm in ("N1", "N2", "N3", "N4"):
        clone(make_normalizer(norm, (1, 3, 2)))
    with pytest.raises(ValueError):
        make_normalizer("N5", (1, 3, 2))
