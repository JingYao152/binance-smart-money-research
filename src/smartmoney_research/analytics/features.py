"""Causal market indicators and strict as-of trade-event alignment."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    rsi_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    volume_window: int = 20
    volatility_window: int = 20

    def __post_init__(self):
        for name in ("rsi_period", "ema_fast", "ema_slow", "macd_signal", "atr_period", "volume_window", "volatility_window"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be smaller than ema_slow")


def compute_indicators(ohlcv: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Calculate indicators using only current and earlier market bars.

    Exponential averages use ``adjust=False`` and expose their usual warm-up as
    NaN. ATR is a rolling mean of true range. Trend strength is the absolute
    fast/slow EMA spread scaled by ATR; it is a derived scale-free trend feature,
    not an exchange-provided signal.
    """
    settings = config or FeatureConfig()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV is missing columns: {sorted(missing)}")
    sort_column = "open_time" if "open_time" in ohlcv.columns else "close_time" if "close_time" in ohlcv.columns else None
    if sort_column is None:
        raise ValueError("OHLCV requires open_time or close_time for chronological ordering")

    result = ohlcv.copy().sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    for column in ("open", "high", "low", "close", "volume"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    close = result["close"]

    result["ema_fast"] = close.ewm(span=settings.ema_fast, adjust=False, min_periods=settings.ema_fast).mean()
    result["ema_slow"] = close.ewm(span=settings.ema_slow, adjust=False, min_periods=settings.ema_slow).mean()
    result["macd"] = result["ema_fast"] - result["ema_slow"]
    result["macd_signal"] = result["macd"].ewm(span=settings.macd_signal, adjust=False,
                                                 min_periods=settings.macd_signal).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / settings.rsi_period, adjust=False,
                         min_periods=settings.rsi_period).mean()
    avg_loss = losses.ewm(alpha=1 / settings.rsi_period, adjust=False,
                          min_periods=settings.rsi_period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi"] = 100 - 100 / (1 + relative_strength)
    result.loc[(avg_gain > 0) & (avg_loss == 0), "rsi"] = 100.0
    result.loc[(avg_gain == 0) & (avg_loss == 0) & avg_gain.notna() & avg_loss.notna(), "rsi"] = 50.0

    previous_close = close.shift(1)
    true_range = pd.concat(
        [result["high"] - result["low"],
         (result["high"] - previous_close).abs(),
         (result["low"] - previous_close).abs()], axis=1,
    ).max(axis=1, skipna=True)
    result["atr"] = true_range.rolling(settings.atr_period, min_periods=settings.atr_period).mean()

    volume = result["volume"]
    result["volume_mean"] = volume.rolling(settings.volume_window, min_periods=settings.volume_window).mean()
    result["volume_std"] = volume.rolling(settings.volume_window, min_periods=settings.volume_window).std(ddof=1)
    result["volume_zscore"] = ((volume - result["volume_mean"]) / result["volume_std"].replace(0, np.nan))
    result["trend_strength"] = (result["ema_fast"] - result["ema_slow"]).abs() / result["atr"].replace(0, np.nan)
    returns = close.pct_change(fill_method=None)
    result["rolling_volatility"] = returns.rolling(settings.volatility_window,
                                                     min_periods=settings.volatility_window).std(ddof=1)
    return result


def align_trade_events(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Join each event to the last bar whose close_time is strictly earlier.

    ``event_time`` is an epoch timestamp in the same unit as ``close_time``.
    Bars with an equal close time are deliberately excluded to avoid using a
    value that was not yet available at the event timestamp.
    """
    if "event_time" not in events.columns:
        raise ValueError("events require event_time")
    if "close_time" not in bars.columns:
        raise ValueError("bars require close_time")
    if events.empty:
        result = events.copy()
        if "bar_close_time" not in result.columns:
            result["bar_close_time"] = pd.Series(dtype=bars["close_time"].dtype)
        return result
    if bars.empty:
        raise ValueError("events have no prior fully closed bar")

    left = events.copy()
    left["__event_order"] = np.arange(len(left))
    left = left.sort_values("event_time", kind="mergesort")
    right = bars.copy().sort_values("close_time", kind="mergesort")
    event_has_close_time = "close_time" in left.columns
    rename = {column: f"bar_{column}" for column in right.columns
              if column == "close_time" or column in left.columns}
    right = right.rename(columns=rename)
    right_time = rename.get("close_time", "close_time")
    aligned = pd.merge_asof(left, right, left_on="event_time", right_on=right_time,
                            direction="backward", allow_exact_matches=False,
                            suffixes=("", "_bar"))
    if right_time != "bar_close_time":
        aligned = aligned.rename(columns={right_time: "bar_close_time"})
    if aligned["bar_close_time"].isna().any():
        missing_count = int(aligned["bar_close_time"].isna().sum())
        raise ValueError(f"{missing_count} event(s) have no prior fully closed bar")
    if not (aligned["bar_close_time"] < aligned["event_time"]).all():
        raise ValueError("aligned bars must close strictly before each event")
    if not event_has_close_time:
        aligned["close_time"] = aligned["bar_close_time"]
    return aligned.sort_values("__event_order", kind="mergesort").drop(columns="__event_order").reset_index(drop=True)
