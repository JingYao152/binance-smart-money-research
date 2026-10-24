import numpy as np
import pandas as pd
import pytest

from smartmoney_research.analytics.features import FeatureConfig, align_trade_events, compute_indicators


def test_indicators_use_documented_series_and_keep_warmup_missing():
    bars = pd.DataFrame({
        "open_time": [0, 60_000, 120_000, 180_000],
        "close_time": [59_999, 119_999, 179_999, 239_999],
        "open": [100.0, 101.0, 102.0, 101.0],
        "high": [101.0, 102.0, 103.0, 102.0],
        "low": [99.0, 100.0, 101.0, 99.0],
        "close": [100.0, 101.0, 102.0, 101.0],
        "volume": [10.0, 20.0, 30.0, 40.0],
    })
    config = FeatureConfig(rsi_period=2, ema_fast=2, ema_slow=3, macd_signal=2,
                           atr_period=2, volume_window=2, volatility_window=2)

    result = compute_indicators(bars, config)

    assert np.isnan(result.loc[0, "ema_fast"])
    assert result.loc[1, "ema_fast"] == pytest.approx(100.6666667)
    assert result.loc[2, "rsi"] == pytest.approx(100.0)
    assert result.loc[3, "rsi"] < 100.0
    assert result.loc[1, "atr"] == pytest.approx(2.0)
    assert result.loc[3, "atr"] == pytest.approx(2.5)
    assert result.loc[1, "volume_mean"] == pytest.approx(15.0)
    assert result.loc[1, "volume_zscore"] == pytest.approx(0.70710678)
    assert np.isnan(result.loc[0, "rolling_volatility"])
    assert result.loc[2, "rolling_volatility"] == pytest.approx(np.std([0.01, 1 / 101], ddof=1))
    assert result.loc[2, "trend_strength"] > 0
    assert {"macd", "macd_signal", "macd_histogram"} <= set(result.columns)


def test_indicators_sort_bars_chronologically_before_calculation():
    bars = pd.DataFrame({
        "open_time": [60_000, 0], "close_time": [119_999, 59_999],
        "open": [11.0, 10.0], "high": [12.0, 11.0], "low": [10.0, 9.0],
        "close": [11.0, 10.0], "volume": [2.0, 1.0],
    })

    result = compute_indicators(bars, FeatureConfig(ema_fast=1, ema_slow=2, rsi_period=1,
                                                    atr_period=1, macd_signal=1,
                                                    volume_window=1, volatility_window=1))

    assert result["close_time"].tolist() == [59_999, 119_999]
    assert result["close"].tolist() == [10.0, 11.0]


def test_trade_alignment_uses_only_bars_closed_strictly_before_event():
    bars = pd.DataFrame({"close_time": [100, 200, 300], "close": [10.0, 20.0, 30.0]})
    events = pd.DataFrame({"event_id": ["same-close", "after-close"], "event_time": [200, 201]})

    aligned = align_trade_events(events, bars)

    assert aligned["event_id"].tolist() == ["same-close", "after-close"]
    assert aligned["close_time"].tolist() == [100, 200]
    assert aligned["close"].tolist() == [10.0, 20.0]
    assert (aligned["close_time"] < aligned["event_time"]).all()


def test_trade_alignment_rejects_events_without_a_prior_closed_bar():
    bars = pd.DataFrame({"close_time": [100], "close": [10.0]})
    events = pd.DataFrame({"event_id": ["too-early"], "event_time": [100]})

    with pytest.raises(ValueError, match="no prior fully closed bar"):
        align_trade_events(events, bars)


def test_feature_config_rejects_invalid_periods():
    with pytest.raises(ValueError, match="ema_fast"):
        FeatureConfig(ema_fast=0)
    with pytest.raises(ValueError, match="ema_fast must be smaller"):
        FeatureConfig(ema_fast=20, ema_slow=10)


def test_rsi_reports_neutral_value_for_a_flat_series_after_warmup():
    bars = pd.DataFrame({
        "open_time": [0, 1, 2], "close": [10.0, 10.0, 10.0],
        "open": [10.0, 10.0, 10.0], "high": [11.0, 11.0, 11.0],
        "low": [9.0, 9.0, 9.0], "volume": [1.0, 1.0, 1.0],
    })

    result = compute_indicators(bars, FeatureConfig(rsi_period=2, ema_fast=1, ema_slow=2,
                                                    macd_signal=1, atr_period=2,
                                                    volume_window=2, volatility_window=2))

    assert result.loc[2, "rsi"] == pytest.approx(50.0)
