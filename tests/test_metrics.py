import pandas as pd
import pytest

from smartmoney_research.analytics.metrics import analyze_positions


def _positions():
    return pd.DataFrame([
        {"record_id": "win-a", "side": "LONG", "status": "CLOSED", "open_time": 1_000,
         "close_time": 3_000, "opened_volume": 10.0, "closed_volume": 10.0,
         "net_pnl": 100.0, "gross_pnl": 110.0, "funding_pnl": -5.0,
         "trading_fee": 5.0, "liquidation_fee": 0.0, "leverage": 5.0,
         "unrealized_pnl": None},
        {"record_id": "win-partial", "side": "SHORT", "status": "CLOSED", "open_time": 2_000,
         "close_time": 6_000, "opened_volume": 10.0, "closed_volume": 6.0,
         "net_pnl": 50.0, "gross_pnl": 52.0, "funding_pnl": 0.0,
         "trading_fee": 2.0, "liquidation_fee": 0.0, "leverage": 3.0,
         "unrealized_pnl": None},
        {"record_id": "loss", "side": "LONG", "status": "CLOSED", "open_time": 1_000,
         "close_time": 2_000, "opened_volume": 5.0, "closed_volume": 5.0,
         "net_pnl": -50.0, "gross_pnl": -48.0, "funding_pnl": 0.0,
         "trading_fee": 2.0, "liquidation_fee": 0.0, "leverage": 2.0,
         "unrealized_pnl": None},
        {"record_id": "open", "side": "LONG", "status": "OPEN", "open_time": 5_000,
         "close_time": None, "opened_volume": 4.0, "closed_volume": 1.0,
         "net_pnl": None, "gross_pnl": None, "funding_pnl": None,
         "trading_fee": None, "liquidation_fee": None, "leverage": 10.0,
         "unrealized_pnl": 8.0},
    ])


def test_position_metrics_separate_closed_results_from_open_positions_and_label_partials():
    summary = analyze_positions(_positions())

    assert summary["position_count"] == 4
    assert summary["closed_position_count"] == 3
    assert summary["open_position_count"] == 1
    assert summary["pnl_sample_count"] == 3
    assert summary["win_rate"] == pytest.approx(2 / 3)
    assert summary["profit_factor"] == pytest.approx(3.0)
    assert summary["payoff_ratio"] == pytest.approx(1.5)
    assert summary["realized_pnl"] == pytest.approx(100.0)
    assert summary["funding_pnl"] == pytest.approx(-5.0)
    assert summary["fees_total"] == pytest.approx(9.0)
    assert summary["average_holding_ms"] == pytest.approx(7_000 / 3)
    assert summary["side_counts"] == {"LONG": 3, "SHORT": 1}
    assert summary["partial_close_count"] == 2
    assert summary["partial_position_ids"] == ["open", "win-partial"]
    assert summary["open_unrealized_pnl"] == pytest.approx(8.0)
    assert summary["max_drawdown_pct"] is None
    assert summary["drawdown_status"] == "missing_equity_series"


def test_zero_loss_sample_does_not_report_infinite_profit_factor_or_payoff():
    positions = _positions().iloc[:2].copy()
    result = analyze_positions(positions)

    assert result["profit_factor"] is None
    assert result["payoff_ratio"] is None
    assert result["profit_factor_status"] == "no_losses_in_sample"


def test_drawdown_uses_explicit_positive_equity_series_only():
    chart = pd.DataFrame({"timestamp": [1, 2, 3, 4], "equity": [100.0, 120.0, 90.0, 110.0]})
    result = analyze_positions(_positions(), chart)
    pnl_chart = pd.DataFrame({"timestamp": [1, 2, 3], "value": [100.0, 120.0, 90.0]})

    assert result["max_drawdown_pct"] == pytest.approx(0.25)
    assert result["drawdown_status"] == "observed_equity_series"
    assert analyze_positions(_positions(), pnl_chart)["max_drawdown_pct"] is None


def test_missing_fee_and_funding_fields_remain_unverifiable():
    positions = _positions().iloc[:1].copy()
    positions.loc[0, "trading_fee"] = None
    positions.loc[0, "funding_pnl"] = None
    result = analyze_positions(positions)

    assert result["fees_total"] is None
    assert result["funding_pnl"] is None
    assert "trading_fee" in result["missing_fields"]
    assert "funding_pnl" in result["missing_fields"]


def test_per_symbol_outcomes_keep_known_samples_and_missing_pnl_explicit():
    positions = pd.DataFrame([
        {"record_id": "syn-win", "symbol": "SYNUSDT", "status": "CLOSED", "close_time": 2_000, "net_pnl": 10.0},
        {"record_id": "syn-missing", "symbol": "SYNUSDT", "status": "CLOSED", "close_time": 3_000, "net_pnl": None},
        {"record_id": "btc-loss", "symbol": "BTCUSDT", "status": "CLOSED", "close_time": 4_000, "net_pnl": -5.0},
    ])

    result = analyze_positions(positions)
    synthetic = result["per_symbol"]["SYNUSDT"]
    bitcoin = result["per_symbol"]["BTCUSDT"]

    assert synthetic["position_count"] == 2
    assert synthetic["closed_position_count"] == 2
    assert synthetic["pnl_sample_count"] == 1
    assert synthetic["pnl_missing_count"] == 1
    assert synthetic["realized_pnl"] == pytest.approx(10.0)
    assert synthetic["realized_pnl_complete"] is False
    assert synthetic["win_rate"] == pytest.approx(1.0)
    assert synthetic["missing_fields"]["net_pnl"] == 1
    assert bitcoin["position_count"] == 1
    assert bitcoin["realized_pnl"] == pytest.approx(-5.0)
    assert bitcoin["win_rate"] == pytest.approx(0.0)


def test_drawdown_is_unavailable_when_any_equity_point_is_missing_or_invalid():
    chart = pd.DataFrame({"timestamp": [1, None, 3, 4, 5],
                          "equity": [100.0, 110.0, 90.0, 0.0, None]})

    result = analyze_positions(_positions(), chart)

    assert result["max_drawdown_pct"] is None
    assert result["drawdown_status"] == "partial_equity_series"
    assert result["drawdown_missing_point_count"] == 3


def test_drawdown_rejects_duplicate_point_timestamps():
    chart = pd.DataFrame({"timestamp": [1, 1, 3], "equity": [100.0, 120.0, 90.0]})

    result = analyze_positions(_positions(), chart)

    assert result["max_drawdown_pct"] is None
    assert result["drawdown_status"] == "partial_equity_series"
    assert result["drawdown_missing_point_count"] == 2
