import numpy as np
import pandas as pd
import pytest

from smartmoney_research.backtest import (
    BacktestConfig,
    run_backtest,
    run_sensitivity,
    run_walk_forward,
)
from smartmoney_research.strategies import StrategyRule


def _bars(closes, opens=None, *, symbol="SYNUSDT"):
    opens = list(closes) if opens is None else opens
    frame = pd.DataFrame({
        "open_time": np.arange(len(closes), dtype=np.int64) * 60_000,
        "close_time": np.arange(len(closes), dtype=np.int64) * 60_000 + 59_999,
        "open": opens,
        "high": [max(o, c) + 0.5 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 0.5 for o, c in zip(opens, closes)],
        "close": closes,
        "volume": [100.0] * len(closes),
    })
    frame.attrs["symbol"] = symbol
    return frame


def _cross_up_bars():
    closes = [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 8.0, 7.0]
    opens = [10.1, 9.1, 8.1, 9.1, 10.1, 12.0, 10.2, 8.2, 7.2]
    return _bars(closes, opens)


def test_signal_from_closed_bar_fills_at_the_next_bar_open():
    bars = _cross_up_bars()
    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, fee_rate=0, slippage_rate=0))

    trade = result["trades"][0]
    assert trade["signal_index"] == 4
    assert trade["entry_index"] == 5
    assert trade["entry_time"] == bars.loc[5, "open_time"]
    assert trade["entry_price"] == pytest.approx(bars.loc[5, "open"])
    assert trade["entry_time"] > bars.loc[trade["signal_index"], "close_time"]
    assert trade["exit_index"] == 8
    assert trade["exit_time"] == bars.loc[8, "open_time"]
    assert trade["exit_time_exact"] is True
    assert trade["exit_price"] == pytest.approx(bars.loc[8, "open"])


def test_fees_and_slippage_reduce_final_equity():
    bars = _cross_up_bars()
    rule = StrategyRule("ema_crossover", fast=2, slow=3)
    baseline = run_backtest(bars, pd.DataFrame(), rule,
                            BacktestConfig(initial_cash=10_000, fee_rate=0, slippage_rate=0))
    costly = run_backtest(bars, pd.DataFrame(), rule,
                          BacktestConfig(initial_cash=10_000, fee_rate=0.002, slippage_rate=0.003))

    assert costly["equity"].iloc[-1] < baseline["equity"].iloc[-1]
    assert costly["assumptions"]["fee_rate"] == pytest.approx(0.002)
    assert costly["assumptions"]["slippage_rate"] == pytest.approx(0.003)


@pytest.mark.parametrize("direction, expected_sign", [("long", -1), ("short", 1)])
def test_signed_funding_only_applies_while_position_is_open(direction, expected_sign):
    if direction == "long":
        closes = [10, 9, 8, 9, 10, 11, 10, 8, 7]
    else:
        closes = [10, 11, 12, 11, 10, 9, 10, 12, 13]
    bars = _bars([float(value) for value in closes])
    # For the short case, invert the price path so the 2/3 EMA crosses downward.
    rule = StrategyRule("ema_crossover", fast=2, slow=3, direction=direction)
    funding = pd.DataFrame({
        "symbol": ["SYNUSDT"] * 3,
        "funding_time": [bars.loc[1, "open_time"], bars.loc[5, "open_time"], bars.loc[8, "open_time"]],
        "funding_rate": [0.01, 0.01, 0.01],
        "mark_price": [10.0, 10.0, 10.0],
    })

    result = run_backtest(bars, funding, rule, BacktestConfig(initial_cash=10_000, fee_rate=0, slippage_rate=0))

    assert result["funding"]["rows_used"] == 1
    assert result["funding"]["rows_skipped_position_closed"] == 2
    assert np.sign(result["funding"]["funding_pnl"]) == expected_sign
    assert result["trades"][0]["funding_pnl"] == pytest.approx(result["funding"]["funding_pnl"])


def test_funding_inside_intrabar_stop_exit_bar_is_counted_as_ambiguous():
    bars = _cross_up_bars()
    bars.loc[6, "low"] = 5.0
    funding = pd.DataFrame({
        "symbol": ["SYNUSDT"],
        "funding_time": [bars.loc[6, "open_time"] + 1_000],
        "funding_rate": [0.01],
        "mark_price": [10.0],
    })

    result = run_backtest(bars, funding, StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, leverage=2, stop_loss_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    trade = result["trades"][0]
    assert trade["exit_time"] is None
    assert trade["exit_time_exact"] is False
    assert trade["exit_time_interval"]["start"] == bars.loc[6, "open_time"]
    assert trade["exit_time_interval"]["end"] == bars.loc[6, "close_time"]
    assert result["funding"]["rows_skipped_intrabar_exit_ambiguous"] == 1
    assert result["funding"]["rows_used"] == 0
    assert result["funding"]["funding_pnl"] == 0


def test_missing_funding_mark_uses_only_a_completed_prior_bar_price():
    bars = _bars([10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 8.0, 7.0])
    event_time = int(bars.loc[6, "open_time"] + 1_000)
    funding = pd.DataFrame({
        "symbol": ["SYNUSDT"],
        "funding_time": [event_time],
        "funding_rate": [0.01],
    })

    result = run_backtest(bars, funding, StrategyRule("ema_crossover", fast=2, slow=3, direction="long"),
                          BacktestConfig(initial_cash=10_000, fee_rate=0, slippage_rate=0))

    expected = -result["trades"][0]["size"] * bars.loc[5, "close"] * 0.01
    assert result["funding"]["funding_pnl"] == pytest.approx(expected)
    assert result["funding"]["rows_missing_mark_price"] == 1
    assert result["funding"]["missing_price_fallbacks"]["last_completed_close"] == 1


def test_missing_prior_close_falls_back_to_the_known_entry_fill():
    bars = _bars([10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 10.0, 8.0, 7.0]).drop(columns="close_time")
    funding = pd.DataFrame({
        "symbol": ["SYNUSDT"],
        "funding_time": [bars.loc[5, "open_time"]],
        "funding_rate": [0.01],
    })

    result = run_backtest(bars, funding, StrategyRule("ema_crossover", fast=2, slow=3, direction="long"),
                          BacktestConfig(initial_cash=10_000, fee_rate=0, slippage_rate=0))

    expected = -result["trades"][0]["size"] * result["trades"][0]["entry_price"] * 0.01
    assert result["funding"]["funding_pnl"] == pytest.approx(expected)
    assert result["funding"]["missing_price_fallbacks"]["trade_entry_price"] == 1


def test_leverage_defaults_to_one_and_cannot_exceed_two():
    assert BacktestConfig().leverage == pytest.approx(1.0)
    assert BacktestConfig().max_leverage == pytest.approx(2.0)
    assert BacktestConfig(leverage=2.0).leverage == pytest.approx(2.0)
    with pytest.raises(ValueError, match="2x"):
        BacktestConfig(leverage=2.01)
    with pytest.raises(ValueError, match="max_leverage"):
        BacktestConfig(max_leverage=2.01)


@pytest.mark.parametrize('direction', ['long', 'short'])
@pytest.mark.parametrize('leverage', [1.3, 2.0])
@pytest.mark.parametrize('slippage', [.002, .1])
def test_every_entry_respects_own_equity_leverage_after_losses_and_funding(monkeypatch, direction, leverage, slippage):
    # Close signals at rows 0/2/4/6 produce two losing round trips, then an open trade.
    signals = pd.DataFrame({
        'long_entries': [True, False, False, False, True, False, False, False, True, False],
        'long_exits': [False, False, True, False, False, False, True, False, False, False],
        'short_entries': [False] * 10, 'short_exits': [False] * 10,
    })
    if direction == 'short':
        signals.columns = ['short_entries', 'short_exits', 'long_entries', 'long_exits']
    monkeypatch.setattr('smartmoney_research.backtest.build_signals', lambda *args: signals)
    prices = [100., 100., 100., 80., 100., 100., 100., 80., 100., 100.]
    if direction == 'short':
        prices[3] = prices[7] = 120.
    bars = _bars(prices)
    funding = pd.DataFrame({'funding_time': [90_000, 330_000],
                            'funding_rate': [.02 if direction == 'long' else -.02] * 2,
                            'mark_price': [100., 100.]})
    config = BacktestConfig(leverage=leverage, max_leverage=leverage, fee_rate=.001, slippage_rate=slippage)
    result = run_backtest(bars, funding, StrategyRule('ema_crossover', direction=direction), config)
    assert len(result['trades']) == 3
    own_equity = config.initial_cash
    for trade in result['trades']:
        notional = trade['entry_price'] * trade['size']
        equity_after_fee = own_equity - trade['entry_fees']
        slippage_cost = abs(trade['entry_price'] - prices[trade['entry_index']]) * trade['size']
        assert notional / (equity_after_fee - slippage_cost) <= config.max_leverage + 1e-12
        if trade['status'] == 'closed':
            assert trade['pnl'] < 0
            assert trade['funding_pnl'] < 0
            own_equity += trade['pnl']
    last = result['trades'][-1]
    assert last['own_equity_before_entry'] == pytest.approx(own_equity)
    assert last['entry_notional'] == pytest.approx(last['entry_price'] * last['size'])
    last_slippage_cost = abs(last['entry_price'] - prices[last['entry_index']]) * last['size']
    assert last['entry_leverage'] == pytest.approx(last['entry_notional'] / (own_equity - last['entry_fees'] - last_slippage_cost))
    expected_final = own_equity - last['entry_fees'] + (
        1 if direction == 'long' else -1) * last['size'] * (prices[-1] - last['entry_price'])
    assert result['equity'].iloc[-1] == pytest.approx(expected_final)
    assert result['funding']['funding_pnl'] == pytest.approx(sum(t['funding_pnl'] for t in result['trades']))


def test_nonpositive_own_equity_prevents_subsequent_entries(monkeypatch):
    signals = pd.DataFrame({
        'long_entries': [True, False, False, False, True, False],
        'long_exits': [False, False, True, False, False, False],
        'short_entries': [False] * 6, 'short_exits': [False] * 6,
    })
    monkeypatch.setattr('smartmoney_research.backtest.build_signals', lambda *args: signals)
    bars = _bars([100., 100., 100., 20., 100., 100.])
    result = run_backtest(bars, pd.DataFrame(), StrategyRule('ema_crossover'),
                          BacktestConfig(leverage=2, fee_rate=0, slippage_rate=0))
    assert len(result['trades']) == 1
    assert result['equity'].iloc[-1] < 0


def test_walk_forward_windows_are_chronological_and_sensitivity_is_finite():
    closes = [100 + np.sin(index / 3) * 5 + index * 0.1 for index in range(120)]
    bars = _bars(closes)
    rule = StrategyRule("ema_crossover", fast=2, slow=5)

    windows = run_walk_forward(bars, pd.DataFrame(), rule, BacktestConfig(),
                               train_rows=60, test_rows=20, step_rows=20,
                               parameter_grid={"fast": [2, 3], "slow": [5]})
    sensitivity = run_sensitivity(bars.iloc[:60], pd.DataFrame(), rule,
                                  {"fast": [2, 3], "slow": [5]}, BacktestConfig())

    assert windows
    assert all(window["train_end_time"] < window["test_start_time"] for window in windows)
    assert all(window["test_end_time"] <= bars["close_time"].iloc[-1] for window in windows)
    assert all(window["selection_source"].startswith("training_") for window in windows)
    assert all(item["evaluated_through"] <= window["train_end_time"]
               for window in windows for item in window["training_grid"]["results"])
    assert all(len(window["out_of_sample_sensitivity"]) == 2 for window in windows)
    assert all(item["evaluated_through"] == window["test_end_time"]
               for window in windows for item in window["out_of_sample_sensitivity"])
    assert len(sensitivity["results"]) == 2
    assert "selected_parameters" not in sensitivity
    assert all(item["evaluated_through"] <= bars["close_time"].iloc[59] for item in sensitivity["results"])


def test_sensitivity_rejects_a_grid_with_invalid_parameter_combinations():
    bars = _bars([100.0 + np.sin(index / 3) for index in range(30)])

    with pytest.raises(ValueError, match="invalid parameter combination"):
        run_sensitivity(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=5),
                        {"fast": [2, 10], "slow": [5]}, BacktestConfig())


def test_ohlc_high_low_drives_stop_loss_exit_and_liquidation_is_labeled_estimate():
    bars = _cross_up_bars()
    bars.loc[6, "low"] = 5.0
    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, leverage=2, stop_loss_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    assert any(trade["exit_reason"] == "stop_loss" for trade in result["trades"])
    assert result["liquidation_risk"]["basis"] == "estimate"


def test_stop_precedes_a_same_bar_next_open_signal_exit_when_ohlc_hits_stop():
    closes = [100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 100.5, 100.0, 100.2]
    bars = _bars(closes)
    bars.loc[8, "low"] = 80.0

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, stop_loss_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    trade = result["trades"][0]
    assert trade["exit_index"] == 8
    assert trade["exit_reason"] == "stop_loss"
    assert result["assumptions"]["stop_vs_signal_priority"] == (
        "An OHLC stop or take-profit hit takes precedence over a close-derived exit on the same next bar; "
        "the signal exit fills at the open only when no stop is triggered."
    )


def test_liquidation_estimate_excludes_exit_bar_extremes_after_stop_fill():
    bars = _cross_up_bars()
    bars.loc[6, "low"] = 5.0

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, leverage=2, stop_loss_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    trade = result["trades"][0]
    assert trade["exit_reason"] == "stop_loss"
    assert trade["liquidation_risk_estimate"] is False
    assert trade["maximum_adverse_excursion_pct"] < result["liquidation_risk"]["estimated_threshold_pct"]
    assert "modeled exit fill" in result["liquidation_risk"]["measurement_basis"]


def test_ohlc_high_triggers_take_profit_when_close_stays_below_target():
    bars = _bars([10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    bars.loc[6, "high"] = 12.5

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, take_profit_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    trade = result["trades"][0]
    assert trade["status"] == "closed"
    assert trade["exit_reason"] == "take_profit"
    assert trade["exit_price"] == pytest.approx(trade["entry_price"] * 1.1)


def test_vectorbt_starts_stop_checks_after_the_entry_bar_and_labels_that_assumption():
    bars = _bars([10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    bars.loc[5, "low"] = 1.0

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(initial_cash=10_000, stop_loss_pct=0.1,
                                         fee_rate=0, slippage_rate=0))

    assert result["trades"][0]["status"] == "open"
    assert result["assumptions"]["stop_processing"] == "OHLC stop checks start on the bar after entry"


def test_btc_benchmark_uses_the_same_timestamps_as_the_strategy():
    bars = _cross_up_bars()
    bitcoin = _bars([100.0 + i for i in range(len(bars))], symbol="BTCUSDT")

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(fee_rate=0, slippage_rate=0), benchmark_ohlcv=bitcoin)

    assert result["benchmark"]["available"] is True
    assert result["benchmark"]["same_timestamps"] is True
    assert list(result["benchmark"]["equity"].index) == list(result["equity"].index)
    assert result["benchmark"]["total_return"] == pytest.approx(0.08)
    assert result["benchmark"]["leverage"] == 1.0
    assert result["benchmark"]["costs_included"] is False


def test_non_btc_benchmark_is_never_labeled_as_btc():
    bars = _cross_up_bars()
    another_asset = _bars([100.0 + i for i in range(len(bars))], symbol="ETHUSDT")

    result = run_backtest(bars, pd.DataFrame(), StrategyRule("ema_crossover", fast=2, slow=3),
                          BacktestConfig(fee_rate=0, slippage_rate=0), benchmark_ohlcv=another_asset)

    assert result["benchmark"]["available"] is False
    assert result["benchmark"]["type"] == "BTC buy-and-hold"
    assert "BTCUSDT or BTCUSD_PERP" in result["benchmark"]["reason"]
