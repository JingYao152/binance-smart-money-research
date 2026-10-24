"""Small, auditable strategy rules for the research backtester."""
from dataclasses import asdict, dataclass
import math

import pandas as pd

from smartmoney_research.analytics.features import FeatureConfig, compute_indicators


@dataclass(frozen=True)
class StrategyRule:
    """An allow-listed rule definition; arbitrary expressions are not accepted."""

    kind: str
    direction: str = "long"
    fast: int = 12
    slow: int = 26
    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    exit_level: float = 50.0

    def __post_init__(self) -> None:
        if self.kind not in {"ema_crossover", "rsi_threshold"}:
            raise ValueError("kind must be one of: ema_crossover, rsi_threshold")
        if self.direction not in {"long", "short"}:
            raise ValueError("direction must be long or short; run each side separately")
        if type(self.fast) is not int or self.fast <= 0:
            raise ValueError("fast must be a positive integer")
        if type(self.slow) is not int or self.slow <= 0:
            raise ValueError("slow must be a positive integer")
        if type(self.rsi_period) is not int or self.rsi_period <= 0:
            raise ValueError("rsi_period must be a positive integer")
        for name in ("oversold", "overbought", "exit_level"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be a finite numeric threshold")
        if self.kind == "ema_crossover":
            if self.fast >= self.slow:
                raise ValueError("EMA crossover requires fast < slow")
            if (self.rsi_period, self.oversold, self.overbought, self.exit_level) != (14, 30.0, 70.0, 50.0):
                raise ValueError("RSI parameters are unsupported for ema_crossover")
        else:
            if (self.fast, self.slow) != (12, 26):
                raise ValueError("EMA parameters are unsupported for rsi_threshold")
            if not (0 <= self.oversold < self.overbought <= 100):
                raise ValueError("RSI thresholds require 0 <= oversold < overbought <= 100")
            if not (0 <= self.exit_level <= 100):
                raise ValueError("exit_level must be between 0 and 100")

    def to_dict(self) -> dict:
        return asdict(self)


def build_signals(ohlcv: pd.DataFrame, rule: StrategyRule) -> pd.DataFrame:
    """Return close-derived long/short entry and exit signals.

    The signals are indexed in chronological order and are not executable until
    the following bar. The backtest applies that one-bar execution delay.
    """
    if not isinstance(rule, StrategyRule):
        raise TypeError("rule must be a StrategyRule")
    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        raise ValueError("ohlcv must be a non-empty DataFrame")

    feature_config = FeatureConfig(
        ema_fast=rule.fast if rule.kind == "ema_crossover" else 12,
        ema_slow=rule.slow if rule.kind == "ema_crossover" else 26,
        rsi_period=rule.rsi_period if rule.kind == "rsi_threshold" else 14,
    )
    features = compute_indicators(ohlcv, feature_config)
    false = pd.Series(False, index=features.index, dtype=bool)
    long_entries, long_exits = false.copy(), false.copy()
    short_entries, short_exits = false.copy(), false.copy()

    if rule.kind == "ema_crossover":
        fast, slow = features["ema_fast"], features["ema_slow"]
        crossed_up = fast.gt(slow) & fast.shift(1).le(slow.shift(1))
        crossed_down = fast.lt(slow) & fast.shift(1).ge(slow.shift(1))
        crossed_up = crossed_up.fillna(False)
        crossed_down = crossed_down.fillna(False)
        if rule.direction == "long":
            long_entries, long_exits = crossed_up, crossed_down
        else:
            short_entries, short_exits = crossed_down, crossed_up
    else:
        rsi = features["rsi"]
        previous = rsi.shift(1)
        recovering = rsi.gt(rule.oversold) & previous.le(rule.oversold)
        long_exit = rsi.ge(rule.exit_level) & previous.lt(rule.exit_level)
        cooling = rsi.lt(rule.overbought) & previous.ge(rule.overbought)
        short_exit = rsi.le(rule.exit_level) & previous.gt(rule.exit_level)
        if rule.direction == "long":
            long_entries, long_exits = recovering.fillna(False), long_exit.fillna(False)
        else:
            short_entries, short_exits = cooling.fillna(False), short_exit.fillna(False)

    return pd.DataFrame({
        "long_entries": long_entries.astype(bool),
        "long_exits": long_exits.astype(bool),
        "short_entries": short_entries.astype(bool),
        "short_exits": short_exits.astype(bool),
    })
