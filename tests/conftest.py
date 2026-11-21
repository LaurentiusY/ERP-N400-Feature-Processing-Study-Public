import numpy as np
import pytest

from n400fp.synthetic import make_subject


@pytest.fixture(scope="session")
def subject():
    """Synthetic subject with a clear N400-like effect (120 trials)."""
    return make_subject(n_trials=120, effect_uv=4.0, seed=1)


@pytest.fixture(scope="session")
def null_subject():
    """Synthetic subject without any effect."""
    return make_subject(n_trials=120, effect_uv=0.0, theta_gain=0.0, seed=2, subject="sub-002")


@pytest.fixture(autouse=True)
def _main_preprocessing(monkeypatch):
    """Tests run on the main preprocessing unless they set ``N400_PREPROC`` themselves."""
    monkeypatch.delenv("N400_PREPROC", raising=False)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
