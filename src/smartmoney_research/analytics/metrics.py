"""Source-faithful trader position summaries."""
import math

import pandas as pd


_CLOSED_STATUSES = {"CLOSED", "CLOSE", "FINISHED", "LIQUIDATED", "LIQUIDATION"}


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(_series(frame, name), errors="coerce")


def _known_sum(values: pd.Series):
    known = values.dropna()
    return float(known.sum()) if len(known) else None


def _complete_sum(values: pd.Series, expected_count: int):
    if expected_count == 0 or len(values) != expected_count or values.isna().any():
        return None
    return float(values.sum())


def _drawdown(chart_points: pd.DataFrame | None):
    if chart_points is None:
        return None, "missing_equity_series", 0
    if "equity" not in chart_points.columns or "timestamp" not in chart_points.columns:
        return None, "missing_equity_series", len(chart_points)
    if chart_points.empty:
        return None, "missing_equity_series", 0
    points = chart_points[["timestamp", "equity"]].copy()
    points["timestamp"] = pd.to_numeric(points["timestamp"], errors="coerce")
    points["equity"] = pd.to_numeric(points["equity"], errors="coerce")
    valid = points["timestamp"].notna() & points["equity"].notna()
    valid &= points["timestamp"].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid &= points["equity"].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    valid &= (points["timestamp"] >= 0) & (points["equity"] > 0)
    valid &= ~points["timestamp"].duplicated(keep=False)
    missing_point_count = int((~valid).sum())
    if missing_point_count:
        status = "partial_equity_series" if bool(valid.any()) else "invalid_equity_series"
        return None, status, missing_point_count
    points = points.sort_values("timestamp", kind="mergesort")
    equity = points["equity"]
    if len(equity) < 2:
        return None, "insufficient_equity_points", 0
    drawdown = equity / equity.cummax() - 1
    return float(-drawdown.min()), "observed_equity_series", 0


def _symbol_summaries(frame: pd.DataFrame, closed_mask: pd.Series, net_pnl: pd.Series) -> tuple[dict, int]:
    source_symbols = _series(frame, "symbol")
    text_symbols = source_symbols.astype("string").str.strip().str.upper()
    valid_symbol = source_symbols.notna() & text_symbols.ne("")
    symbol_missing_count = int((~valid_symbol).sum())
    summaries = {}
    for symbol in sorted(str(value) for value in text_symbols.loc[valid_symbol].dropna().unique()):
        symbol_mask = valid_symbol & text_symbols.eq(symbol)
        symbol_closed = closed_mask & symbol_mask
        known = net_pnl.loc[symbol_closed & net_pnl.notna()]
        wins = int((known > 0).sum())
        losses = int((known < 0).sum())
        breakeven = int((known == 0).sum())
        closed_count = int(symbol_closed.sum())
        symbol_rows = frame.loc[symbol_mask]
        missing = {}
        for field in ("net_pnl", "gross_pnl", "funding_pnl", "trading_fee", "liquidation_fee"):
            values = _numeric(symbol_rows, field).loc[symbol_closed]
            missing_count = int(values.isna().sum())
            if missing_count:
                missing[field] = missing_count
        summaries[symbol] = {
            "position_count": int(symbol_mask.sum()),
            "closed_position_count": closed_count,
            "pnl_sample_count": int(len(known)),
            "pnl_missing_count": closed_count - len(known),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "realized_pnl": _known_sum(known),
            "realized_pnl_complete": closed_count > 0 and len(known) == closed_count,
            "win_rate": wins / len(known) if len(known) else None,
            "missing_fields": missing,
        }
    return summaries, symbol_missing_count


def analyze_positions(positions: pd.DataFrame, chart_points: pd.DataFrame | None = None) -> dict:
    """Summarize closed source PnL separately from current unrealized PnL.

    Missing components are never treated as zero. PnL totals over partially
    observed samples include the known sum and expose coverage counts; totals
    that would require replacing unknown fees or funding with zero are ``None``.
    Drawdown is derived only from an explicit positive ``equity`` series.
    """
    frame = positions.copy().reset_index(drop=True)
    count = len(frame)
    close_time = _series(frame, "close_time")
    status = _series(frame, "status").fillna("").astype(str).str.upper()
    closed_mask = close_time.notna() | status.isin(_CLOSED_STATUSES)
    open_mask = ~closed_mask
    closed = frame.loc[closed_mask]
    open_positions = frame.loc[open_mask]

    net_pnl = _numeric(frame, "net_pnl")
    known_outcomes = closed_mask & net_pnl.notna()
    outcomes = net_pnl.loc[known_outcomes]
    wins = int((outcomes > 0).sum())
    losses = int((outcomes < 0).sum())
    breakeven = int((outcomes == 0).sum())
    winning_pnl = outcomes.loc[outcomes > 0]
    losing_pnl = outcomes.loc[outcomes < 0]
    gross_wins = float(winning_pnl.sum()) if len(winning_pnl) else 0.0
    gross_losses = float(-losing_pnl.sum()) if len(losing_pnl) else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else None
    average_win = float(winning_pnl.mean()) if len(winning_pnl) else None
    average_loss = float(-losing_pnl.mean()) if len(losing_pnl) else None
    payoff_ratio = average_win / average_loss if average_win is not None and average_loss else None

    known_net = net_pnl.loc[closed_mask & net_pnl.notna()]
    gross_pnl = _numeric(frame, "gross_pnl").loc[closed_mask]
    funding = _numeric(frame, "funding_pnl").loc[closed_mask]
    trading_fee = _numeric(frame, "trading_fee").loc[closed_mask]
    liquidation_fee = _numeric(frame, "liquidation_fee").loc[closed_mask]
    fees = trading_fee + liquidation_fee
    missing_fields = []
    for field, series in (("net_pnl", known_net), ("gross_pnl", gross_pnl), ("funding_pnl", funding),
                          ("trading_fee", trading_fee), ("liquidation_fee", liquidation_fee)):
        if series.isna().any() or (len(closed) and len(series.dropna()) != len(closed)):
            missing_fields.append(field)

    opened_at = _numeric(frame, "open_time").loc[closed_mask]
    closed_at = _numeric(frame, "close_time").loc[closed_mask]
    holding = (closed_at - opened_at).dropna()
    if (holding < 0).any():
        holding = holding.loc[holding >= 0]

    sides = _series(frame, "side").dropna().astype(str).str.upper()
    side_counts = {}
    for side in sides:
        normalized = "LONG" if side in {"LONG", "BUY"} else "SHORT" if side in {"SHORT", "SELL"} else side
        side_counts[normalized] = side_counts.get(normalized, 0) + 1

    leverage = _numeric(frame, "leverage").dropna()
    leverage_distribution = {}
    for value, amount in leverage.value_counts().sort_index().items():
        leverage_distribution[f"{float(value):g}x"] = int(amount)

    opened_volume = _numeric(frame, "opened_volume")
    closed_volume = _numeric(frame, "closed_volume")
    partial_mask = opened_volume.notna() & closed_volume.notna() & (closed_volume > 0) & (closed_volume < opened_volume)
    record_ids = _series(frame, "record_id")
    if record_ids.isna().all():
        record_ids = _series(frame, "position_id")
    partial_ids = sorted(str(value) for value in record_ids.loc[partial_mask].dropna().unique())

    open_unrealized = _numeric(frame, "unrealized_pnl").loc[open_mask]
    equity_drawdown, drawdown_status, drawdown_missing_count = _drawdown(chart_points)
    per_symbol, symbol_missing_count = _symbol_summaries(frame, closed_mask, net_pnl)
    if not len(outcomes):
        profit_factor_status = "no_known_outcomes"
    elif not losses:
        profit_factor_status = "no_losses_in_sample"
    else:
        profit_factor_status = "computed"

    return {
        "position_count": count,
        "closed_position_count": int(closed_mask.sum()),
        "open_position_count": int(open_mask.sum()),
        "pnl_sample_count": int(len(outcomes)),
        "pnl_missing_count": int(closed_mask.sum() - len(outcomes)),
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate": wins / len(outcomes) if len(outcomes) else None,
        "profit_factor": profit_factor,
        "profit_factor_status": profit_factor_status,
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": payoff_ratio,
        "realized_pnl": _known_sum(known_net),
        "realized_pnl_complete": len(known_net) == len(closed) and len(closed) > 0,
        "gross_pnl": _complete_sum(gross_pnl, len(closed)),
        "funding_pnl": _complete_sum(funding, len(closed)),
        "funding_pnl_known_sum": _known_sum(funding),
        "fees_total": _complete_sum(fees, len(closed)),
        "known_fees_total": _known_sum(fees),
        "average_holding_ms": float(holding.mean()) if len(holding) else None,
        "median_holding_ms": float(holding.median()) if len(holding) else None,
        "side_counts": side_counts,
        "leverage_distribution": leverage_distribution,
        "average_leverage": float(leverage.mean()) if len(leverage) else None,
        "median_leverage": float(leverage.median()) if len(leverage) else None,
        "partial_close_count": len(partial_ids),
        "partial_position_ids": partial_ids,
        "open_unrealized_pnl": _complete_sum(open_unrealized, len(open_positions)),
        "open_unrealized_pnl_known_sum": _known_sum(open_unrealized),
        "per_symbol": per_symbol,
        "symbol_missing_position_count": symbol_missing_count,
        "max_drawdown_pct": equity_drawdown,
        "drawdown_status": drawdown_status,
        "drawdown_missing_point_count": drawdown_missing_count,
        "missing_fields": missing_fields,
    }
