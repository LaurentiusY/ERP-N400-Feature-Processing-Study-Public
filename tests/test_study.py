import pytest

from n400fp.study import (
    BASELINE,
    FACTORS,
    Config,
    full_factorial,
    model_configs,
    pilot,
    stage_configs,
)


def test_full_factorial_is_640_unique():
    configs = full_factorial()
    assert len(configs) == 8 * 4 * 4 * 5 == 640
    assert len({c.code for c in configs}) == 640


def test_lda_has_480_configurations_without_n2():
    lda = model_configs(full_factorial(), "lda")
    assert len(lda) == 480
    assert all(c.N != "N2" for c in lda)


def test_pilot_changes_one_factor_each():
    configs = pilot()
    assert len(configs) == 18
    base = Config.parse(BASELINE)
    assert configs[0] == base
    for c in configs[1:]:
        assert sum(a != b for a, b in zip(c, base, strict=True)) == 1
    assert len({c.code for c in configs}) == 18


def test_parse_roundtrip_and_errors():
    assert Config.parse("B8-W3-N4-C5").code == "B8-W3-N4-C5"
    with pytest.raises(ValueError):
        Config.parse("B9-W1-N1-C1")
    with pytest.raises(ValueError):
        Config.parse("B1-W1-N1")


def test_stages():
    assert [c.code for c in stage_configs("baseline")] == [BASELINE]
    assert len(stage_configs("pilot")) == 18
    assert len(stage_configs("full")) == 640
    assert set(FACTORS) == {"B", "W", "N", "C"}
