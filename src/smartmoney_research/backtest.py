"""Cost-aware chronological backtests backed by VectorBT."""
from dataclasses import asdict, dataclass
from itertools import product
import math
from typing import Any

import numpy as np
import pandas as pd

from smartmoney_research.strategies import StrategyRule, build_signals


_MAX_LEVERAGE = 2.0
_MAX_GRID_COMBINATIONS = 256
_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    leverage: float = 1.0
    max_leverage: float = _MAX_LEVERAGE
    position_size: float = 0.95
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0002
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    maintenance_margin_rate: float = 0.005

    def __post_init__(self) -> None:
        for name in ("initial_cash", "leverage", "max_leverage", "position_size",
                     "fee_rate", "slippage_rate", "maintenance_margin_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.max_leverage < 1 or self.max_leverage > _MAX_LEVERAGE:
            raise ValueError("max_leverage must be between 1x and 2x")
        if self.leverage < 1 or self.leverage > self.max_leverage:
            raise ValueError("leverage must be between 1x and max_leverage, capped at 2x")
        if not (0 < self.position_size <= 0.95):
            raise ValueError("position_size must be in (0, 0.95] to reserve cash for costs")
        if not (0 <= self.fee_rate < 1):
            raise ValueError("fee_rate must be in [0, 1)")
        if not (0 <= self.slippage_rate < 1):
            raise ValueError("slippage_rate must be in [0, 1)")
        if not (0 <= self.maintenance_margin_rate < 1):
            raise ValueError("maintenance_margin_rate must be in [0, 1)")
        for name in ("stop_loss_pct", "take_profit_pct"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 < value < 1):
                raise ValueError(f"{name} must be None or a finite fraction in (0, 1)")


def _ordered_bars(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, str, pd.Index]:
    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        raise ValueError("ohlcv must be a non-empty DataFrame")
    missing = _OHLCV_COLUMNS - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV is missing columns: {sorted(missing)}")
    time_col = "open_time" if "open_time" in ohlcv.columns else "close_time" if "close_time" in ohlcv.columns else None
    if time_col is None:
        raise ValueError("OHLCV requires open_time or close_time")
    bars = ohlcv.copy()
    for timestamp_col in ("open_time", "close_time"):
        if timestamp_col in bars.columns:
            bars[timestamp_col] = pd.to_numeric(bars[timestamp_col], errors="coerce")
    bars = bars.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    if bars[time_col].isna().any() or bars[time_col].duplicated().any():
        raise ValueError(f"{time_col} must contain unique timestamps")
    for timestamp_col in ("open_time", "close_time"):
        if timestamp_col in bars.columns and bars[timestamp_col].isna().any():
            raise ValueError(f"{timestamp_col} must not contain missing timestamps")
    if "open_time" in bars.columns and "close_time" in bars.columns:
        if (bars["close_time"] <= bars["open_time"]).any():
            raise ValueError("close_time must be after open_time")
        if bars["close_time"].duplicated().any() or not bars["close_time"].is_monotonic_increasing:
            raise ValueError("close_time must be unique and chronological")
        if len(bars) > 1 and (bars["open_time"].iloc[1:].to_numpy()
                              <= bars["close_time"].iloc[:-1].to_numpy()).any():
            raise ValueError("OHLCV bars must not overlap")
    for col in _OHLCV_COLUMNS:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    if not np.isfinite(bars[list(_OHLCV_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV prices and volume must be finite")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any() or (bars["volume"] < 0).any():
        raise ValueError("OHLC prices must be positive and volume must be non-negative")
    if (bars["high"] < bars[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high must be at least open, low, and close")
    if (bars["low"] > bars[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low must be at most open, high, and close")
    index = pd.Index(bars[time_col].tolist(), name=time_col)
    if not index.is_monotonic_increasing:
        raise ValueError("OHLCV timestamps must be chronological")
    bars.attrs.update(ohlcv.attrs)
    if "symbol" in bars.columns:
        symbols = bars["symbol"].dropna().astype(str).str.upper().unique().tolist()
        if len(symbols) > 1:
            raise ValueError("one-symbol OHLCV is required for a backtest")
        if symbols:
            attr_symbol = str(bars.attrs.get("symbol", symbols[0])).upper()
            if attr_symbol != symbols[0]:
                raise ValueError("OHLCV symbol column and symbol attribute disagree")
            bars.attrs["symbol"] = symbols[0]
    return bars, time_col, index


def _to_timestamp_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value
    return value


def _time_value(bars: pd.DataFrame, index: int, primary: str) -> Any:
    if primary in bars.columns:
        column = primary
    elif "close_time" in bars.columns:
        column = "close_time"
    elif "open_time" in bars.columns:
        column = "open_time"
    else:
        raise ValueError("OHLCV requires open_time or close_time")
    return _to_timestamp_value(bars.iloc[index][column])


def _portfolio_trades(portfolio: Any, bars: pd.DataFrame, config: BacktestConfig,
                      time_col: str) -> list[dict[str, Any]]:
    records = portfolio.trades.records
    trades: list[dict[str, Any]] = []
    if records.empty:
        return trades
    for row in records.to_dict("records"):
        entry_idx = int(row["entry_idx"])
        exit_raw = row.get("exit_idx")
        is_open = int(row.get("status", 1)) == 0 or pd.isna(exit_raw)
        exit_idx = None if is_open else int(exit_raw)
        direction = "short" if int(row.get("direction", 0)) == 1 else "long"
        entry_price = float(row["entry_price"])
        size = abs(float(row["size"]))
        entry_time = _time_value(bars, entry_idx, time_col)
        bar_exit_time = None if exit_idx is None else _time_value(bars, exit_idx, time_col)
        trade = {
            "signal_index": max(0, entry_idx - 1),
            "entry_index": entry_idx,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "size": size,
            "direction": direction,
            "exit_index": exit_idx,
            "exit_time": bar_exit_time,
            "exit_time_exact": None if is_open else True,
            "exit_time_basis": "open_position" if is_open else "bar_open_fill",
            "exit_time_interval": None,
            "exit_price": None if is_open else float(row["exit_price"]),
            "status": "open" if is_open else "closed",
            "pnl": None if is_open else float(row["pnl"]),
            "return": None if is_open else float(row["return"]),
            "entry_fees": float(row["entry_fees"]),
            "exit_fees": 0.0 if is_open else float(row["exit_fees"]),
            "funding_pnl": 0.0,
            "exit_reason": "open" if is_open else "signal",
        }
        trade["exit_reason"] = _classify_exit(trade, bars, config)
        if trade["exit_reason"] in {"stop_loss", "take_profit"}:
            assert exit_idx is not None
            interval_start = _time_value(bars, exit_idx, "open_time")
            if "close_time" in bars.columns:
                interval_end = _time_value(bars, exit_idx, "close_time")
                interval_end_inclusive = True
            elif exit_idx + 1 < len(bars) and time_col == "open_time":
                interval_end = _time_value(bars, exit_idx + 1, "open_time")
                interval_end_inclusive = False
            else:
                interval_end = interval_start
                interval_end_inclusive = True
            trade["exit_time"] = None
            trade["exit_time_exact"] = False
            trade["exit_time_basis"] = "intrabar_exit_bounded_by_bar"
            trade["exit_time_interval"] = {
                "start": interval_start,
                "end": interval_end,
                "end_inclusive": interval_end_inclusive,
            }
        trades.append(trade)
    return trades


def _classify_exit(trade: dict[str, Any], bars: pd.DataFrame, config: BacktestConfig) -> str:
    if trade["exit_index"] is None:
        return "open"
    if config.stop_loss_pct is None and config.take_profit_pct is None:
        return "signal"
    entry_idx = trade["entry_index"]
    exit_idx = trade["exit_index"]
    entry = trade["entry_price"]
    long = trade["direction"] == "long"
    stop = None if config.stop_loss_pct is None else entry * (
        1 - config.stop_loss_pct if long else 1 + config.stop_loss_pct
    )
    target = None if config.take_profit_pct is None else entry * (
        1 + config.take_profit_pct if long else 1 - config.take_profit_pct
    )
    # Intrabar ordering is unknowable from OHLC. When both thresholds hit, classify
    # the conservative stop first, matching the simulator's stop priority.
    for i in range(entry_idx + 1, exit_idx + 1):
        candle = bars.iloc[i]
        stop_hit = stop is not None and (
            (candle["open"] <= stop or candle["low"] <= stop) if long
            else (candle["open"] >= stop or candle["high"] >= stop)
        )
        if stop_hit:
            return "stop_loss"
        target_hit = target is not None and (
            (candle["open"] >= target or candle["high"] >= target) if long
            else (candle["open"] <= target or candle["low"] <= target)
        )
        if target_hit:
            return "take_profit"
    return "signal"


def _matching_funding(funding: pd.DataFrame, bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if funding is None or funding.empty:
        return pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"]), 0
    if not isinstance(funding, pd.DataFrame):
        raise TypeError("funding must be a DataFrame")
    missing = {"funding_time", "funding_rate"} - set(funding.columns)
    if missing:
        raise ValueError(f"funding is missing columns: {sorted(missing)}")
    rows = funding.copy()
    skipped_symbol = 0
    if "symbol" in rows.columns:
        bar_symbol = bars.attrs.get("symbol")
        symbols = rows["symbol"].dropna().astype(str).unique().tolist()
        if bar_symbol:
            matches = rows["symbol"].astype(str).str.upper() == str(bar_symbol).upper()
            skipped_symbol = int((~matches).sum())
            rows = rows.loc[matches].copy()
        elif len(symbols) > 1:
            raise ValueError("OHLCV symbol is required when funding contains multiple symbols")
    rows["funding_time"] = pd.to_numeric(rows["funding_time"], errors="coerce")
    rows["funding_rate"] = pd.to_numeric(rows["funding_rate"], errors="coerce")
    if rows[["funding_time", "funding_rate"]].isna().any().any():
        raise ValueError("funding_time and funding_rate must be numeric and present")
    if not np.isfinite(rows[["funding_time", "funding_rate"]].to_numpy(dtype=float)).all():
        raise ValueError("funding_time and funding_rate must be finite")
    if "mark_price" not in rows.columns:
        rows["mark_price"] = np.nan
    else:
        rows["mark_price"] = pd.to_numeric(rows["mark_price"], errors="coerce")
    return rows.sort_values("funding_time", kind="mergesort").reset_index(drop=True), skipped_symbol


def _apply_funding(funding: pd.DataFrame, skipped_symbol: int, bars: pd.DataFrame,
                   time_col: str, index: pd.Index, trades: list[dict[str, Any]], equity: pd.Series) -> tuple[pd.Series, dict]:
    rows_used = skipped_closed = skipped_window = skipped_intrabar_ambiguous = missing_price = 0
    funding_pnl = 0.0
    funding_by_trade: dict[int, float] = {}
    missing_price_fallbacks = {"last_completed_close": 0, "trade_entry_price": 0}
    cashflow_by_index = np.zeros(len(bars), dtype=float)
    times = bars[time_col].to_numpy()
    close_times = bars["close_time"].to_numpy() if "close_time" in bars.columns else None
    start_time = times[0]
    end_time = bars["close_time"].iloc[-1] if "close_time" in bars.columns else times[-1]
    for _, row in funding.iterrows():
        timestamp = row["funding_time"]
        if timestamp < start_time or timestamp > end_time:
            skipped_window += 1
            continue
        matching_trade = None
        ambiguous_exit = False
        for i, trade in enumerate(trades):
            if timestamp < trade["entry_time"]:
                continue
            interval = trade.get("exit_time_interval")
            if interval is not None:
                inside_interval = timestamp >= interval["start"] and (
                    timestamp <= interval["end"] if interval["end_inclusive"]
                    else timestamp < interval["end"]
                )
                if inside_interval:
                    ambiguous_exit = True
                    break
                if timestamp < interval["start"]:
                    matching_trade = (i, trade)
                    break
                continue
            exit_time = trade["exit_time"]
            if trade["status"] == "open" or (exit_time is not None and timestamp < exit_time):
                matching_trade = (i, trade)
                break
        if ambiguous_exit:
            skipped_intrabar_ambiguous += 1
            continue
        if matching_trade is None:
            skipped_closed += 1
            continue
        trade_idx, trade = matching_trade
        bar_idx = int(np.searchsorted(times, timestamp, side="right") - 1)
        bar_idx = max(0, min(bar_idx, len(bars) - 1))
        mark_price = row["mark_price"]
        if pd.isna(mark_price) or not np.isfinite(mark_price) or mark_price <= 0:
            # Never use the containing bar's close: it may occur after the
            # funding event. Prefer the most recent completed close strictly
            # before the event, then the already-known entry fill.
            prior_close_idx = -1
            if close_times is not None:
                prior_close_idx = int(np.searchsorted(close_times, timestamp, side="left") - 1)
            if prior_close_idx >= 0:
                mark_price = float(bars.iloc[prior_close_idx]["close"])
                missing_price_fallbacks["last_completed_close"] += 1
            else:
                mark_price = float(trade["entry_price"])
                missing_price_fallbacks["trade_entry_price"] += 1
            missing_price += 1
        position_sign = 1.0 if trade["direction"] == "long" else -1.0
        payment = -position_sign * trade["size"] * float(mark_price) * float(row["funding_rate"])
        funding_by_trade[trade_idx] = funding_by_trade.get(trade_idx, 0.0) + payment
        cashflow_by_index[bar_idx] += payment
        funding_pnl += payment
        rows_used += 1
    for trade_idx, amount in funding_by_trade.items():
        trades[trade_idx]["funding_pnl"] += amount
        if trades[trade_idx]["pnl"] is not None:
            trades[trade_idx]["pnl"] += amount
    adjusted_equity = equity + pd.Series(cashflow_by_index, index=index).cumsum()
    summary = {
        "rows_received": int(len(funding) + skipped_symbol),
        "rows_used": rows_used,
        "rows_skipped_position_closed": skipped_closed,
        "rows_skipped_intrabar_exit_ambiguous": skipped_intrabar_ambiguous,
        "rows_skipped_outside_window": skipped_window,
        "rows_skipped_other_symbol": skipped_symbol,
        "rows_missing_mark_price": missing_price,
        "missing_price_assumption": "use the last completed close strictly before fundingTime, or the trade entry price when none exists",
        "missing_price_fallbacks": missing_price_fallbacks,
        "intrabar_exit_funding_assumption": "Funding events inside a stop or take-profit exit bar are skipped because OHLC cannot order the event and exit.",
        "funding_pnl": float(funding_pnl),
    }
    return adjusted_equity, summary


def _periods_per_year(bars: pd.DataFrame, time_col: str) -> float:
    times = pd.to_numeric(bars[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(times) < 2:
        return 365.25
    spacing = float(np.median(np.diff(times)))
    if spacing <= 0:
        return 365.25
    # Binance epoch timestamps are milliseconds. Synthetic relative timestamps
    # use the same unit; epoch seconds and nanosecond datetime values are also
    # recognized by their scale.
    magnitude = float(np.max(np.abs(times)))
    units_per_year_ms = 365.25 * 24 * 60 * 60 * 1000
    if magnitude >= 1e14:
        units_per_year = units_per_year_ms * 1_000_000
    elif magnitude >= 1e11 or spacing >= 1e4:
        units_per_year = units_per_year_ms
    else:
        units_per_year = units_per_year_ms / 1000
    return max(1.0, units_per_year / spacing)


def _metrics(equity: pd.Series, trades: list[dict[str, Any]], periods_per_year: float,
             initial_cash: float) -> dict[str, Any]:
    equity = pd.to_numeric(equity, errors="coerce").astype(float)
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    if equity.empty:
        return {"total_return": None, "sharpe": None, "sortino": None, "profit_factor": None,
                "win_rate": None, "max_drawdown": None, "closed_trades": 0, "open_trades": 0}
    returns = equity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    mean_return = float(returns.mean()) if len(returns) else 0.0
    volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = returns.clip(upper=0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sharpe = None if volatility == 0 else mean_return / volatility * math.sqrt(periods_per_year)
    sortino = None if downside_deviation == 0 else mean_return / downside_deviation * math.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1.0
    closed = [t for t in trades if t["status"] == "closed" and t["pnl"] is not None]
    pnls = np.asarray([t["pnl"] for t in closed], dtype=float)
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    profit_factor = None if not len(losers) else float(winners.sum() / abs(losers.sum()))
    return {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "sharpe": None if sharpe is None else float(sharpe),
        "sortino": None if sortino is None else float(sortino),
        "profit_factor": profit_factor,
        "win_rate": None if not len(closed) else float((pnls > 0).mean()),
        "max_drawdown": float(drawdown.min()),
        "closed_trades": int(len(closed)),
        "open_trades": int(sum(t["status"] == "open" for t in trades)),
        "winning_trades": int(len(winners)),
        "losing_trades": int(len(losers)),
    }


def _liquidation_summary(trades: list[dict[str, Any]], bars: pd.DataFrame,
                         config: BacktestConfig) -> dict[str, Any]:
    threshold = max(0.0, 1.0 / config.leverage - config.maintenance_margin_rate)
    hits = 0
    for trade in trades:
        exit_idx = trade["exit_index"]
        if exit_idx is None:
            candles = bars.iloc[trade["entry_index"]:]
            trade["liquidation_excursion_basis"] = "full OHLC through the last observed bar while position remains open"
        else:
            # The modeled exit is known, while the rest of its OHLC candle may
            # contain prices after the position has closed. Use pre-exit bars
            # plus the actual modeled fill and exclude that candle's extremes.
            candles = bars.iloc[trade["entry_index"]:exit_idx]
            trade["liquidation_excursion_basis"] = "prior full OHLC plus modeled exit fill; exit-bar extremes excluded"
        entry = trade["entry_price"]
        if trade["direction"] == "long":
            prior_adverse = 0.0 if candles.empty else float((entry - candles["low"].min()) / entry)
            fill_adverse = 0.0 if trade["exit_price"] is None else (entry - trade["exit_price"]) / entry
            trade["liquidation_price_estimate"] = entry * (1.0 - threshold)
        else:
            prior_adverse = 0.0 if candles.empty else float((candles["high"].max() - entry) / entry)
            fill_adverse = 0.0 if trade["exit_price"] is None else (trade["exit_price"] - entry) / entry
            trade["liquidation_price_estimate"] = entry * (1.0 + threshold)
        adverse = max(0.0, float(prior_adverse), float(fill_adverse))
        trade["maximum_adverse_excursion_pct"] = adverse
        trade["liquidation_risk_estimate"] = adverse >= threshold
        hits += int(trade["liquidation_risk_estimate"])
    return {
        "basis": "estimate",
        "estimated_threshold_pct": float(threshold),
        "trades_at_or_beyond_estimate": hits,
        "formula": "max(0, 1 / leverage - maintenance_margin_rate)",
        "measurement_basis": "Closed trades use full OHLC before exit and the modeled exit fill; exit-bar extremes after the fill are excluded. Open trades use all observed OHLC through the last bar.",
        "limitations": "Does not model exchange maintenance tiers, mark-price paths, or liquidation fees.",
    }


def _benchmark_summary(benchmark_ohlcv: pd.DataFrame | None, bars: pd.DataFrame,
                       time_col: str, initial_cash: float) -> dict[str, Any]:
    symbol = str(bars.attrs.get("symbol", "")).upper()
    benchmark = benchmark_ohlcv
    if benchmark is None and symbol in {"BTCUSDT", "BTCUSD_PERP"}:
        benchmark = bars
    if benchmark is None:
        return {"available": False, "type": "BTC buy-and-hold", "reason": "BTC benchmark bars were not supplied"}
    try:
        btc, btc_time_col, _ = _ordered_bars(benchmark)
    except (TypeError, ValueError) as exc:
        return {"available": False, "type": "BTC buy-and-hold", "reason": str(exc)}
    if time_col not in bars.columns or btc_time_col not in btc.columns:
        return {"available": False, "type": "BTC buy-and-hold", "reason": "timestamps are unavailable"}
    benchmark_symbol = str(btc.attrs.get("symbol", "")).upper()
    if benchmark_symbol not in {"BTCUSDT", "BTCUSD_PERP"}:
        return {
            "available": False,
            "type": "BTC buy-and-hold",
            "reason": "benchmark symbol must be identified as BTCUSDT or BTCUSD_PERP",
        }
    target_times = bars[time_col].tolist()
    btc_lookup = dict(zip(btc[btc_time_col].tolist(), btc["close"].tolist()))
    if not all(time in btc_lookup for time in target_times):
        return {"available": False, "type": "BTC buy-and-hold", "reason": "BTC bars do not cover every strategy timestamp"}
    closes = np.asarray([btc_lookup[time] for time in target_times], dtype=float)
    equity = initial_cash * closes / closes[0]
    return {
        "available": True,
        "type": "BTC buy-and-hold",
        "leverage": 1.0,
        "costs_included": False,
        "costs_assumption": "gross, unlevered buy-and-hold; excludes fees, slippage, and funding",
        "start_time": _to_timestamp_value(target_times[0]),
        "end_time": _to_timestamp_value(target_times[-1]),
        "equity": pd.Series(equity, index=pd.Index(target_times, name=time_col), name="btc_buy_and_hold"),
        "total_return": float(equity[-1] / initial_cash - 1.0),
        "same_timestamps": True,
    }


def _size_trade_segments(trades: list[dict[str, Any]], bars: pd.DataFrame, index: pd.Index,
                         funding: pd.DataFrame, config: BacktestConfig,
                         time_col: str) -> tuple[list[dict[str, Any]], pd.Series]:
    """Size non-overlapping fills against realized own equity, including funding.

    VectorBT establishes unit-quantity signal/stop fills independently of this
    ledger. Prices and percentage costs are quantity-independent, so each segment
    can be scaled once its predecessors' realized cashflows are known.
    """
    # Unit funding is linear in quantity. A closed segment's full funding is
    # known before the next entry; none of its future funding sizes its own entry.
    _apply_funding(funding, 0, bars, time_col, index, trades,
                   pd.Series(0.0, index=index))
    own_equity = config.initial_cash
    equity = np.full(len(bars), config.initial_cash, dtype=float)
    sized = []
    allocation = config.leverage * config.position_size
    for unit in trades:
        if own_equity <= 0:
            break
        trade = unit.copy()
        # Reserve entry fees and the immediate fill-to-open slippage loss.
        # notional / equity_after_entry_costs == allocation <= max_leverage.
        open_price = float(bars['open'].iloc[trade['entry_index']])
        slippage_fraction = abs(trade['entry_price'] - open_price) / trade['entry_price']
        entry_cost_fraction = config.fee_rate + slippage_fraction
        notional = own_equity * allocation / (1 + allocation * entry_cost_fraction)
        quantity = notional / trade['entry_price']
        scale = quantity / unit['size']
        unit_funding = unit['funding_pnl']
        trade.update(size=quantity, funding_pnl=0.0,
                     entry_fees=unit['entry_fees'] * scale,
                     exit_fees=unit['exit_fees'] * scale,
                     own_equity_before_entry=own_equity, entry_notional=notional)
        trade['own_equity_after_entry_fee'] = own_equity - trade['entry_fees']
        trade['entry_slippage_cost'] = notional * slippage_fraction
        trade['own_equity_after_entry_costs'] = trade['own_equity_after_entry_fee'] - trade['entry_slippage_cost']
        trade['entry_leverage'] = notional / trade['own_equity_after_entry_costs']
        if trade['entry_leverage'] > config.max_leverage + 1e-12:
            raise RuntimeError('entry notional exceeds the configured own-equity leverage cap')
        start = trade['entry_index']
        stop = len(bars) if trade['exit_index'] is None else trade['exit_index']
        sign = 1 if trade['direction'] == 'long' else -1
        equity[start:stop] += (sign * quantity * (
            bars['close'].iloc[start:stop].to_numpy() - trade['entry_price']) - trade['entry_fees'])
        if trade['status'] == 'closed':
            trade['pnl'] = (unit['pnl'] - unit_funding) * scale
            equity[stop:] += trade['pnl']
            own_equity += trade['pnl'] + unit_funding * scale
        sized.append(trade)
    return sized, pd.Series(equity, index=index, name='equity')


def _run_backtest(ohlcv: pd.DataFrame, funding: pd.DataFrame, rule: StrategyRule,
                  config: BacktestConfig, benchmark_ohlcv: pd.DataFrame | None = None,
                  first_execution_index: int = 0) -> dict[str, Any]:
    bars, time_col, index = _ordered_bars(ohlcv)
    if not isinstance(rule, StrategyRule):
        raise TypeError("rule must be a StrategyRule")
    if not isinstance(config, BacktestConfig):
        raise TypeError("config must be a BacktestConfig")
    if not 0 <= first_execution_index < len(bars):
        raise ValueError("first_execution_index must identify a bar in ohlcv")

    signals = build_signals(bars, rule)
    # A close is only known after its bar ends. The shifted order arrays execute
    # at the next bar's open, which is passed as VectorBT's order price.
    entries = signals["long_entries"].shift(1, fill_value=False).astype(bool)
    exits = signals["long_exits"].shift(1, fill_value=False).astype(bool)
    short_entries = signals["short_entries"].shift(1, fill_value=False).astype(bool)
    short_exits = signals["short_exits"].shift(1, fill_value=False).astype(bool)
    if first_execution_index:
        entries.iloc[:first_execution_index] = False
        exits.iloc[:first_execution_index] = False
        short_entries.iloc[:first_execution_index] = False
        short_exits.iloc[:first_execution_index] = False

    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError("VectorBT is required for backtests; install the project runtime dependencies") from exc

    kwargs = {
        "close": pd.Series(bars["close"].to_numpy(dtype=float), index=index),
        "entries": pd.Series(entries.to_numpy(dtype=bool), index=index),
        "exits": pd.Series(exits.to_numpy(dtype=bool), index=index),
        "short_entries": pd.Series(short_entries.to_numpy(dtype=bool), index=index),
        "short_exits": pd.Series(short_exits.to_numpy(dtype=bool), index=index),
        "size": 1.0,
        "size_type": "amount",
        "price": pd.Series(bars["open"].to_numpy(dtype=float), index=index),
        "fees": config.fee_rate,
        "slippage": config.slippage_rate,
        # Fully fund the unit-fill schedule, even for repeated short losses.
        # The separate ledger below owns actual capital and position sizing.
        "init_cash": float(bars['high'].sum()) * 10 + 1,
        "upon_opposite_entry": "close",
        "open": pd.Series(bars["open"].to_numpy(dtype=float), index=index),
        "high": pd.Series(bars["high"].to_numpy(dtype=float), index=index),
        "low": pd.Series(bars["low"].to_numpy(dtype=float), index=index),
        "stop_entry_price": "fillprice",
        "stop_exit_price": "stopmarket",
        "upon_stop_exit": "close",
        "cash_sharing": False,
    }
    if config.stop_loss_pct is not None:
        kwargs["sl_stop"] = config.stop_loss_pct
    if config.take_profit_pct is not None:
        kwargs["tp_stop"] = config.take_profit_pct
    portfolio = vbt.Portfolio.from_signals(**kwargs)

    unit_trades = _portfolio_trades(portfolio, bars, config, time_col)
    normalized_funding, skipped_symbol = _matching_funding(funding, bars)
    trades, equity = _size_trade_segments(unit_trades, bars, index, normalized_funding, config, time_col)
    equity, funding_summary = _apply_funding(normalized_funding, skipped_symbol, bars,
                                              time_col, index, trades, equity)
    for trade in trades:
        if trade['pnl'] is not None:
            trade['return'] = trade['pnl'] / (trade['entry_notional'] + trade['entry_fees'])
    periods_per_year = _periods_per_year(bars, time_col)
    metrics = _metrics(equity, trades, periods_per_year, config.initial_cash)
    liquidation = _liquidation_summary(trades, bars, config)
    benchmark = _benchmark_summary(benchmark_ohlcv, bars, time_col, config.initial_cash)
    assumptions = asdict(config)
    assumptions.update({
        "execution": "close-derived signal uses the following bar open unless a stop or take-profit hits in that bar",
        "stop_processing": "OHLC stop checks start on the bar after entry",
        "stop_intrabar_priority": "stop loss is checked before take profit when both levels hit in one bar",
        "stop_vs_signal_priority": "An OHLC stop or take-profit hit takes precedence over a close-derived exit on the same next bar; the signal exit fills at the open only when no stop is triggered.",
        "position_sizing": "position_size times leverage times current own equity after reserving entry fees and fill-to-open slippage loss; prior realized trading costs and funding reduce the next entry budget",
        "leverage_model": "sequential own-equity ledger with entry notional capped against equity after entry costs; no new entries when own equity is nonpositive; borrowing interest excluded",
        "simulation_model": "VectorBT unit-quantity signal/stop fills scaled as non-overlapping trade segments by the own-equity ledger",
        "funding_model": "signed rate times position quantity and mark price; exact bar-open exits use their fill time, while funding inside a stop or take-profit exit bar is marked ambiguous and skipped",
        "funding_symbol_matching": "OHLCV symbol is matched when available; a caller-supplied single-symbol funding frame is assumed to match when OHLCV has no symbol",
        "liquidation_model": "estimate only; closed trades use prior-bar OHLC and the modeled exit fill, excluding exit-bar extremes",
        "benchmark_model": "BTC buy-and-hold is gross and unlevered; fees, slippage, and funding are excluded",
        "vectorbt_version": str(vbt.__version__),
    })
    return {
        "trades": trades,
        "equity": equity,
        "assumptions": assumptions,
        "metrics": metrics,
        "funding": funding_summary,
        "liquidation_risk": liquidation,
        "benchmark": benchmark,
        "periods_per_year": periods_per_year,
    }


def run_backtest(ohlcv: pd.DataFrame, funding: pd.DataFrame, rule: StrategyRule,
                 config: BacktestConfig, *,
                 benchmark_ohlcv: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run a single-scenario VectorBT simulation with explicit costs."""
    return _run_backtest(ohlcv, funding, rule, config, benchmark_ohlcv)


def _grid_rules(rule: StrategyRule, grid: dict[str, list[Any]]) -> list[tuple[dict[str, Any], StrategyRule]]:
    allowed = {"fast", "slow"} if rule.kind == "ema_crossover" else {
        "rsi_period", "oversold", "overbought", "exit_level"
    }
    if not isinstance(grid, dict) or not grid:
        raise ValueError("parameter grid must be a non-empty finite mapping")
    if set(grid) - allowed:
        raise ValueError(f"unsupported grid parameters: {sorted(set(grid) - allowed)}")
    keys = sorted(grid)
    values = [grid[key] for key in keys]
    if any(not isinstance(items, (list, tuple)) or len(items) == 0 for items in values):
        raise ValueError("each grid parameter must have a non-empty list or tuple of values")
    combinations = math.prod(len(items) for items in values)
    if combinations > _MAX_GRID_COMBINATIONS:
        raise ValueError(f"parameter grid exceeds the {_MAX_GRID_COMBINATIONS}-combination limit")
    candidates: list[tuple[dict[str, Any], StrategyRule]] = []
    for combination in product(*values):
        parameters = dict(zip(keys, combination))
        try:
            candidate = StrategyRule(**{**rule.to_dict(), **parameters})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid parameter combination {parameters}: {exc}") from exc
        candidates.append((parameters, candidate))
    if not candidates:
        raise ValueError("parameter grid contains no valid rule combinations")
    return candidates


def run_sensitivity(ohlcv: pd.DataFrame, funding: pd.DataFrame, rule: StrategyRule,
                    parameter_grid: dict[str, list[Any]], config: BacktestConfig,
                    *, benchmark_ohlcv: pd.DataFrame | None = None) -> dict[str, Any]:
    """Evaluate a finite grid and report each run without choosing a winner."""
    bars, _, _ = _ordered_bars(ohlcv)
    results = []
    for parameters, candidate in _grid_rules(rule, parameter_grid):
        result = _run_backtest(bars, funding, candidate, config, benchmark_ohlcv)
        results.append({
            "parameters": parameters,
            "rule": candidate.to_dict(),
            "evaluated_through": _time_value(bars, len(bars) - 1, "close_time"),
            "metrics": result["metrics"],
        })
    return {"results": results, "combinations_evaluated": len(results)}


def _out_of_sample_metrics(simulated: dict[str, Any], context: pd.DataFrame,
                           local_test_index: int, config: BacktestConfig) -> tuple[dict, pd.Series]:
    reference_index = max(0, local_test_index - 1)
    equity = simulated["equity"].iloc[reference_index:].copy()
    trades = [trade for trade in simulated["trades"] if trade["entry_index"] >= local_test_index]
    time_col = "open_time" if "open_time" in context.columns else "close_time"
    periods = _periods_per_year(context.iloc[local_test_index:], time_col)
    return _metrics(equity, trades, periods, config.initial_cash), equity


def run_walk_forward(ohlcv: pd.DataFrame, funding: pd.DataFrame, rule: StrategyRule,
                     config: BacktestConfig, *, train_rows: int, test_rows: int,
                     step_rows: int | None = None,
                     parameter_grid: dict[str, list[Any]] | None = None,
                     selection_metric: str = "sharpe",
                     benchmark_ohlcv: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """Run rolling chronological folds; any parameter choice sees training rows only."""
    bars, _, _ = _ordered_bars(ohlcv)
    for name, value in (("train_rows", train_rows), ("test_rows", test_rows),
                        ("step_rows", test_rows if step_rows is None else step_rows)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if selection_metric not in {"sharpe", "sortino", "total_return", "profit_factor"}:
        raise ValueError("selection_metric is unsupported")
    step = test_rows if step_rows is None else step_rows
    windows: list[dict[str, Any]] = []
    test_start = train_rows
    while test_start < len(bars):
        train_start = max(0, test_start - train_rows)
        test_end = min(len(bars), test_start + test_rows)
        train = bars.iloc[train_start:test_start].copy()
        selected_rule = rule
        selection_source = "fixed_rule"
        train_sensitivity = None
        if parameter_grid is not None:
            train_sensitivity = run_sensitivity(train, funding, rule, parameter_grid, config)
            valid = [item for item in train_sensitivity["results"]
                     if item["metrics"].get(selection_metric) is not None
                     and math.isfinite(float(item["metrics"][selection_metric]))]
            if valid:
                selected = max(valid, key=lambda item: float(item["metrics"][selection_metric]))
                selected_rule = StrategyRule(**selected["rule"])
                selection_source = f"training_{selection_metric}"

        context = bars.iloc[train_start:test_end].copy()
        test_execution_index = test_start - train_start
        simulated = _run_backtest(context, funding, selected_rule, config, benchmark_ohlcv,
                                  first_execution_index=test_execution_index)
        test_metrics, test_equity = _out_of_sample_metrics(
            simulated, context, test_execution_index, config
        )
        out_of_sample_sensitivity = []
        if parameter_grid is not None:
            for parameters, candidate in _grid_rules(rule, parameter_grid):
                candidate_result = _run_backtest(
                    context, funding, candidate, config, benchmark_ohlcv,
                    first_execution_index=test_execution_index,
                )
                candidate_metrics, _ = _out_of_sample_metrics(
                    candidate_result, context, test_execution_index, config
                )
                out_of_sample_sensitivity.append({
                    "parameters": parameters,
                    "rule": candidate.to_dict(),
                    "evaluated_through": _time_value(bars, test_end - 1, "close_time"),
                    "metrics": candidate_metrics,
                })
        train_result = _run_backtest(train, funding, selected_rule, config)
        windows.append({
            "train_start_time": _time_value(bars, train_start, "open_time"),
            "train_end_time": _time_value(bars, test_start - 1, "close_time"),
            "test_start_time": _time_value(bars, test_start, "open_time"),
            "test_end_time": _time_value(bars, test_end - 1, "close_time"),
            "train_metrics": train_result["metrics"],
            "test_metrics": test_metrics,
            "test_equity": test_equity,
            "rule": selected_rule.to_dict(),
            "selection_source": selection_source,
            "training_grid": train_sensitivity,
            "out_of_sample_sensitivity": out_of_sample_sensitivity,
        })
        test_start += step
    return windows
