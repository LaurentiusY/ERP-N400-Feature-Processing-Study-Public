import numpy as np
import pytest


@pytest.mark.eegnet
def test_eegnet_runs_on_baseline(subject):
    pytest.importorskip("torch")
    pytest.importorskip("braindecode")
    from n400fp.config import EEGNetConfig
    from n400fp.eegnet import run_eegnet, series_input
    from n400fp.study import Config

    cfg = Config.parse("B1-W2-N2-C2")
    x = series_input(subject, cfg)
    assert x.shape[:2] == (120, 7)
    out = run_eegnet(subject, [cfg], EEGNetConfig(n_splits=2, seeds=(0,), max_epochs=2))
    assert len(out) == 2 and out.bacc.between(0, 1).all()


def test_series_input_rejects_multi_component(subject):
    from n400fp.eegnet import series_input
    from n400fp.study import Config

    with pytest.raises(ValueError):
        series_input(subject, Config.parse("B6-W1-N1-C1"))
    assert np.isfinite(series_input(subject, Config.parse("B2-W1-N1-C5"))).all()
