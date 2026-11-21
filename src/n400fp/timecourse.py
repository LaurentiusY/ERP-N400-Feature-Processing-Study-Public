"""Time-resolved decoding curve (section 6.5 step 2).

B1, N2, logistic regression with nested C, the subject's outer folds; each 50 ms bin from
0 to 800 ms on its own, with channel sets C1 and C5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DecodeConfig, FeatureConfig
from .decode import evaluate
from .features import SubjectData, apply_channels, bin_average

CHANNEL_SETS = ("C1", "C5")
SPAN = (0.0, 0.8)


def timecourse(
    data: SubjectData,
    splits,
    dcfg: DecodeConfig = DecodeConfig(),
    fcfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    binned = bin_average(data.wave, data.wave_times, SPAN, fcfg.bin_width)  # (n, ch, 16)
    starts = SPAN[0] + fcfg.bin_width * np.arange(binned.shape[-1])
    rows = []
    for code in CHANNEL_SETS:
        x_all, _ = apply_channels(binned, code, data.ch_names)
        for k, start in enumerate(starts):
            x = x_all[:, :, k]  # (n, channels)
            shape = (1, x.shape[1], 1)
            for r in evaluate(x, data.y, splits, "lr", "N2", shape, dcfg):
                rows.append(
                    {
                        "subject": data.subject,
                        "channels": code,
                        "bin_start": round(float(start), 3),
                        "repeat": r["repeat"],
                        "fold": r["fold"],
                        "bacc": r["bacc"],
                        "C_selected": r["C"],
                    }
                )
    return pd.DataFrame(rows)
