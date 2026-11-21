"""EEGNet generality check (section 6.2).

Only representations that are still time series qualify: B1-B5, cut to the window of D2
without bin averaging, channels C1-C5 and normalizations N1-N4 (fitted on the training
fold, with shape ``(1, channels, samples)``). Settings follow Kessler et al.'s
``src/eegnet.py`` (EEGNetv4, 200 epochs, batch 16, class weights, no validation split),
except that the folds are fixed per subject and three seeds are averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EEGNetConfig, FeatureConfig
from .features import SubjectData, apply_channels, component_series
from .normalize import make_normalizer
from .paths import subject_number
from .study import REPRESENTATIONS, WINDOWS, Config

TIME_SERIES_B = ("B1", "B2", "B3", "B4", "B5")


def eegnet_name() -> str:
    """braindecode < 1.0 calls the model EEGNetv4; 1.x renamed it EEGNet (same architecture)."""
    import braindecode.models

    return "EEGNetv4" if hasattr(braindecode.models, "EEGNetv4") else "EEGNet"


def series_input(data: SubjectData, config: Config, fcfg: FeatureConfig = FeatureConfig()):
    if config.B not in TIME_SERIES_B:
        raise ValueError(f"EEGNet needs a single time series representation, not {config.B}")
    (component,) = REPRESENTATIONS[config.B]
    series, times = component_series(data, component, fcfg)
    lo, hi = WINDOWS[config.W]
    mask = (times >= lo - 1e-6) & (times < hi - 1e-6)
    x, _ = apply_channels(series[..., mask], config.C, data.ch_names)
    return x  # (trials, channels, samples)


def run_eegnet(
    data: SubjectData,
    configs: list[Config],
    ecfg: EEGNetConfig = EEGNetConfig(),
    device: str = "cpu",
    sfreq: float = 256.0,
) -> pd.DataFrame:
    import torch
    from braindecode import EEGClassifier
    from braindecode.util import set_random_seeds
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.utils.class_weight import compute_class_weight

    y = data.y
    folds = list(
        StratifiedKFold(
            ecfg.n_splits, shuffle=True, random_state=subject_number(data.subject)
        ).split(np.zeros(len(y)), y)
    )
    model = eegnet_name()
    rows = []
    for config in configs:
        x = series_input(data, config)
        shape = (1, x.shape[1], x.shape[2])
        for seed in ecfg.seeds:
            for k, (tr, te) in enumerate(folds):
                norm = make_normalizer(config.N, shape).fit(x[tr].reshape(len(tr), -1), y[tr])
                xtr = norm.transform(x[tr].reshape(len(tr), -1)).reshape(len(tr), *shape[1:])
                xte = norm.transform(x[te].reshape(len(te), -1)).reshape(len(te), *shape[1:])
                set_random_seeds(seed, cuda=device.startswith("cuda"))
                weights = compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])
                net = EEGClassifier(
                    model,
                    module__n_chans=x.shape[1],
                    module__n_outputs=2,
                    module__n_times=x.shape[2],
                    module__sfreq=sfreq,
                    module__final_conv_length="auto",
                    criterion__weight=torch.tensor(weights, dtype=torch.float32),
                    train_split=None,
                    max_epochs=ecfg.max_epochs,
                    batch_size=ecfg.batch_size,
                    device=device,
                    verbose=0,
                )
                net.fit(xtr.astype(np.float32), y[tr])
                pred = net.predict(xte.astype(np.float32))
                rows.append(
                    {
                        "subject": data.subject,
                        "model": "eegnet",
                        "config": config.code,
                        "B": config.B,
                        "W": config.W,
                        "N": config.N,
                        "C": config.C,
                        "seed": seed,
                        "fold": k,
                        "bacc": float(balanced_accuracy_score(y[te], pred)),
                    }
                )
    return pd.DataFrame(rows)
