import numpy as np

from n400fp.events import behaviour_summary, target_trials


def test_targets_rt_and_correctness():
    onsets = [1.0, 2.1, 2.7, 5.0, 6.2, 9.0, 10.1, 10.5, 12.0, 13.2, 13.9]
    codes = ["111", "211", "201", "122", "222", "112", "212", "boundary", "121", "221", "202"]
    t = target_trials(onsets, codes)
    assert list(t.trigger) == ["211", "222", "212", "221"]
    assert list(t.condition) == ["related", "unrelated", "related", "unrelated"]
    assert list(t.label) == [0, 1, 0, 1]
    assert list(t.event_index) == [0, 1, 2, 3]
    # RT from the delay-corrected target: (2.7 - 2.126) s
    assert np.isclose(t.rt_ms[0], 574.0)
    assert t.correct[0] == 1.0
    # no response before the next prime -> missing
    assert np.isnan(t.rt_ms[1]) and np.isnan(t.correct[1])
    # a boundary event does not end the search, but the next prime does
    assert np.isnan(t.rt_ms[2])
    assert np.isclose(t.rt_ms[3], (13.9 - 13.226) * 1000)
    assert t.correct[3] == 0.0


def test_codes_are_normalized_and_sorted():
    t = target_trials([3.0, 1.0, 1.5], ["221.0", " 211", "201"])
    assert list(t.trigger) == ["211", "221"]
    assert np.isclose(t.rt_ms[0], (1.5 - 1.026) * 1000)


def test_behaviour_summary():
    t = target_trials([1.0, 1.6, 3.0, 3.8], ["211", "201", "221", "201"])
    s = behaviour_summary(t)
    assert s["n_related"] == 1 and s["n_unrelated"] == 1 and s["n_no_response"] == 0
    assert np.isclose(s["median_rt_unrelated"], 774.0)
