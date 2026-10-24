import pandas as pd
import pytest

from smartmoney_research.strategies import StrategyRule, build_signals


def _bars(closes):
    return pd.DataFrame({
        "open_time": range(len(closes)),
        "close_time": range(len(closes)),
        "open": closes,
        "high": [value + 0.5 for value in closes],
        "low": [value - 0.5 for value in closes],
        "close": closes,
        "volume": [1.0] * len(closes),
    })


def test_ema_crossover_is_an_allowlisted_closed_bar_rule():
    rule = StrategyRule(kind="ema_crossover", fast=2, slow=3, direction="long")
    bars = _bars([10, 9, 8, 9, 10, 11, 10, 8, 7])

    signals = build_signals(bars, rule)

    assert signals["long_entries"].iloc[4]
    assert signals["long_exits"].iloc[7]
    assert not signals["short_entries"].any()
    assert not signals["short_exits"].any()


def test_rsi_threshold_rule_supports_short_direction():
    rule = StrategyRule(kind="rsi_threshold", direction="short", rsi_period=2,
                        oversold=30, overbought=70, exit_level=50)
    bars = _bars([10, 11, 12, 11, 10, 9, 10, 12])

    signals = build_signals(bars, rule)

    assert signals["short_entries"].any()
    assert signals["short_exits"].any()
    assert not signals["long_entries"].any()


def test_default_strategy_direction_is_long_only():
    assert StrategyRule("ema_crossover").direction == "long"


@pytest.mark.parametrize("rule", [
    {"kind": "arbitrary_python", "direction": "long"},
    {"kind": "ema_crossover", "fast": 10, "slow": 5},
    {"kind": "rsi_threshold", "oversold": 75, "overbought": 25},
    {"kind": "ema_crossover", "direction": "both"},
])
def test_strategy_rule_rejects_unsupported_or_invalid_values(rule):
    with pytest.raises(ValueError):
        StrategyRule(**rule)
