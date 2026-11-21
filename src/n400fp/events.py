"""Target trials and behaviour from the event list (pure functions, no MNE needed).

Every target word (211/212 related, 221/222 unrelated) is one trial. Its response is the
first response trigger (201 correct, 202 incorrect) after the target and before the next
prime or target; trials without one get a missing reaction time. Other codes (for example
EEGLAB "boundary" events) are ignored. Reaction times are measured from
the delay-corrected target onset (target trigger + 26 ms).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import LABELS, LCD_DELAY, PRIMES, RELATED, RESPONSE_CORRECT, RESPONSES, UNRELATED

TRIAL_COLUMNS = ["event_index", "onset", "trigger", "condition", "label", "rt_ms", "correct"]


def _norm(code: object) -> str:
    """EEGLAB descriptions may come as '211', '211.0' or ' 211'."""
    s = str(code).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def target_trials(
    onsets: Sequence[float], descriptions: Sequence[object], delay: float = LCD_DELAY
) -> pd.DataFrame:
    """One row per target word, in temporal order.

    Columns: ``event_index`` (0-based position among the targets), ``onset`` (s, trigger
    time without the LCD delay), ``trigger``, ``condition`` ("related"/"unrelated"),
    ``label`` (0/1), ``rt_ms`` (float, NaN without response), ``correct`` (1.0/0.0/NaN).
    ``delay``: display delay added to the target time before computing the reaction time
    (ERP CORE: 26 ms; PURSUE: none).
    """
    onsets = np.asarray(onsets, dtype=float)
    codes = [_norm(d) for d in descriptions]
    order = np.argsort(onsets, kind="stable")
    onsets = onsets[order]
    codes = [codes[i] for i in order]

    targets = set(RELATED) | set(UNRELATED)
    stimuli = targets | set(PRIMES)
    rows = []
    for i, (t, code) in enumerate(zip(onsets, codes, strict=True)):
        if code not in targets:
            continue
        condition = "related" if code in RELATED else "unrelated"
        rt, correct = np.nan, np.nan
        for t2, code2 in zip(onsets[i + 1 :], codes[i + 1 :], strict=True):
            if code2 in RESPONSES:
                rt = (t2 - (t + delay)) * 1000.0
                correct = 1.0 if code2 == RESPONSE_CORRECT else 0.0
                break
            if code2 in stimuli:  # the next prime or target ends the search
                break
        rows.append((len(rows), t, code, condition, LABELS[condition], rt, correct))
    return pd.DataFrame(rows, columns=TRIAL_COLUMNS)


def behaviour_summary(trials: pd.DataFrame) -> dict:
    """Per-subject summary used in section 6.5 step 1."""
    out: dict = {"n_trials": int(len(trials)), "n_no_response": int(trials.rt_ms.isna().sum())}
    for cond in ("related", "unrelated"):
        sub = trials[trials.condition == cond]
        ok = sub[sub.correct == 1.0]
        out[f"n_{cond}"] = int(len(sub))
        out[f"accuracy_{cond}"] = float(sub.correct.mean()) if len(sub) else np.nan
        out[f"median_rt_{cond}"] = float(ok.rt_ms.median()) if len(ok) else np.nan
    return out
