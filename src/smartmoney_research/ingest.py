"""Resumable, evidence-preserving ingestion for public trader data."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .clients.binance import BinanceClient
from .clients.http import PaginationError
from .clients.kopyon import KopyonClient
from .normalize import normalize_order, normalize_position

_DAY_MS = 24 * 60 * 60 * 1000
_BASE = "/v5/binance-smart-money"
_BINANCE_MARKET_TYPE = "UM"
_MARKET_COVERAGE_PREFIX = "market_coverage_start_ms="


@dataclass
class SyncReport:
    trader_id: str
    started_at: datetime
    finished_at: datetime | None = None
    complete: bool = False
    resources: dict[str, str] = field(default_factory=dict)
    row_counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["started_at"] = self.started_at.isoformat()
        result["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return result


def _utc(value: datetime | None) -> datetime:
    now = value or datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _timestamp(record: dict, resource: str) -> int | None:
    names = ("openedAt", "closedAt", "updatedAt") if resource == "positionHistory" else ("createdAt", "updatedAt")
    values = []
    for name in names:
        value = record.get(name)
        if value is not None:
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                continue
    return max(values) if values else None


def _checkpoint_watermark(store, trader_id: str, market_type: str, resource: str) -> int | None:
    checkpoint = store.get_checkpoint(trader_id, market_type, resource)
    return checkpoint.get("watermark") if checkpoint else None


def _save_raw(store, endpoint: str, params: dict, payload: Any, fetched_at: datetime) -> str:
    return store.save_raw("kopyon", endpoint, params, payload, 200, fetched_at)


def _save_market_raw(store, endpoint: str, params: dict, payload: Any, fetched_at: datetime) -> str:
    return store.save_raw("binance", endpoint, params, payload, 200, fetched_at)


def _unwrap_page(value):
    if isinstance(value, dict) and isinstance(value.get("data"), dict):
        return value, value["data"]
    return {"data": value}, value


def _market_symbols(store, trader_id: str, market_type: str, configured) -> list[str]:
    symbols = set(configured)
    for table in ("open_positions", "position_history", "order_history"):
        rows = store.connection.execute(
            f"SELECT DISTINCT symbol FROM {table} WHERE trader_id = ? AND market_type = ? AND symbol IS NOT NULL",
            [trader_id, market_type],
        ).fetchall()
        symbols.update(str(row[0]).upper() for row in rows if row[0])
    return sorted(symbols)


def _earliest_trader_event(store, trader_id: str, market_type: str) -> int | None:
    candidates = []
    for table, expression in (("position_history", "COALESCE(open_time, close_time, updated_at)"),
                              ("open_positions", "COALESCE(open_time, updated_at)"),
                              ("order_history", "COALESCE(timestamp, created_at, updated_at)")):
        value = store.connection.execute(
            f"SELECT MIN({expression}) FROM {table} WHERE trader_id = ? AND market_type = ?",
            [trader_id, market_type],
        ).fetchone()[0]
        if value is not None and int(value) >= 0:
            candidates.append(int(value))
    return min(candidates) if candidates else None


def _market_coverage_start(checkpoint) -> int | None:
    """Read the coverage tag stored in cursor for reserved market checkpoints only."""
    if not checkpoint:
        return None
    cursor = checkpoint.get("cursor")
    if not isinstance(cursor, str) or not cursor.startswith(_MARKET_COVERAGE_PREFIX):
        return None
    try:
        return int(cursor[len(_MARKET_COVERAGE_PREFIX):])
    except ValueError:
        return None


def _next_market_coverage_start(previous: int | None, current: int) -> int:
    return current if previous is None else min(previous, current)


def _market_start_ms(prior: int | None, coverage_start: int | None, earliest_event: int | None, all_history: bool,
                     end_ms: int, lookback_days: int, overlap_ms: int) -> int:
    fallback = max(0, end_ms - lookback_days * _DAY_MS)
    if prior is None:
        return earliest_event if all_history and earliest_event is not None else fallback
    if all_history and coverage_start is None:
        # Legacy market checkpoints have a watermark but no proven lower bound.
        return earliest_event if earliest_event is not None else fallback
    if all_history and earliest_event is not None and coverage_start is not None and earliest_event < coverage_start:
        # Newly discovered older trader events extend coverage once.
        return earliest_event
    if prior is not None:
        return max(0, int(prior) - overlap_ms)
    return fallback


def _normalize_kline(symbol: str, interval: str, source: list, raw_id: str, fetched_at: datetime) -> dict:
    if not isinstance(source, list) or len(source) != 12:
        raise ValueError("Expected a 12-field Binance kline")
    return dict(symbol=symbol, interval=interval, open_time=int(source[0]), open=float(source[1]), high=float(source[2]),
                low=float(source[3]), close=float(source[4]), volume=float(source[5]), close_time=int(source[6]),
                quote_volume=float(source[7]), trades=int(source[8]), taker_buy_base_volume=float(source[9]),
                taker_buy_quote_volume=float(source[10]), source="binance", raw_response_id=raw_id,
                fetched_at=fetched_at, raw_fields=source)


def _unfinished_kline_start(store, symbol: str, interval: str) -> int | None:
    """Refresh by candle open, including stale snapshots after their close."""
    value = store.connection.execute(
        '''SELECT MIN(open_time) FROM klines WHERE symbol = ? AND interval = ?
           AND (fetched_at IS NULL OR close_time IS NULL OR close_time >= epoch_ms(fetched_at))''',
        [symbol, interval],
    ).fetchone()[0]
    return None if value is None else int(value)


def _normalize_funding(symbol: str, source: dict, raw_id: str, fetched_at: datetime) -> dict:
    if not isinstance(source, dict) or "fundingTime" not in source or "fundingRate" not in source:
        raise ValueError("Funding-rate row is missing fundingTime or fundingRate")
    result = dict(symbol=source.get("symbol", symbol), funding_time=int(source["fundingTime"]),
                  funding_rate=float(source["fundingRate"]), source="binance", raw_response_id=raw_id,
                  fetched_at=fetched_at, raw_fields=source)
    if source.get("markPrice") not in (None, ""):
        result["mark_price"] = float(source["markPrice"])
    if source.get("rateType") is not None:
        result["rate_type"] = source["rateType"]
    return result


def _validate_market_page_times(times: list[int], start_ms: int, end_ms: int, resource: str):
    if times and (times != sorted(times) or times[0] < start_ms or times[-1] > end_ms):
        raise PaginationError(f"Unordered or out-of-range Binance {resource} page")


def sync_trader(config, trader, store, lookback_days: int = 365, all_history: bool = False, *, kopyon_client=None, binance_client=None, now: datetime | None = None) -> SyncReport:
    """Sync trader records plus Binance market context for relevant symbols.

    Each page is stored before normalization. A failed or hidden resource keeps
    its previous high-water mark so the next run safely overlaps the same data.
    """
    if type(lookback_days) is not int or lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")
    trader_id = trader.get("trader_id", trader.get("traderId")) if isinstance(trader, dict) else str(trader)
    if not trader_id:
        raise ValueError("trader must provide trader_id")
    trader_name = trader.get("name") if isinstance(trader, dict) else None
    started_at = _utc(now)
    end_ms = int(started_at.timestamp() * 1000)
    market_type = config.collection.market_type
    overlap_ms = config.collection.overlap_ms
    report = SyncReport(str(trader_id), started_at)
    if kopyon_client is None:
        http = config.http
        kopyon_client = KopyonClient(api_key=config.kopyon_key, timeout=http.timeout, min_interval=http.min_interval,
                                     max_retries=http.max_retries, backoff_factor=http.backoff_factor)
    if binance_client is None:
        http = config.http
        binance_client = BinanceClient(timeout=http.timeout, min_interval=http.min_interval,
                                       max_retries=http.max_retries, backoff_factor=http.backoff_factor)

    def record_error(resource: str, error: Exception, market: str = market_type):
        previous = _checkpoint_watermark(store, str(trader_id), market, resource)
        store.save_checkpoint(str(trader_id), market, resource, watermark=previous, status="error", error=str(error), updated_at=started_at)
        report.resources[resource] = "error"
        report.errors.append({"resource": resource, "error": str(error)})

    def record_success(resource: str, watermark: int | None = None, market: str = market_type):
        store.save_checkpoint(str(trader_id), market, resource, watermark=watermark, status="complete", updated_at=started_at)
        report.resources[resource] = "complete"

    def record_market_success(resource: str, watermark: int, coverage_start: int):
        # This tagged cursor is reserved for __market_data__ checkpoints. Kopyon
        # history cursors remain provider cursors on trader-scoped resources.
        cursor = f"{_MARKET_COVERAGE_PREFIX}{coverage_start}"
        store.save_checkpoint("__market_data__", _BINANCE_MARKET_TYPE, resource, watermark=watermark,
                             cursor=cursor, status="complete", updated_at=started_at)
        report.resources[resource] = "complete"

    # Store the resolved trader and each profile/chart response before writing normalized snapshots.
    try:
        raw_method = getattr(kopyon_client, "get_profile_response", None)
        if raw_method:
            envelope = raw_method(str(trader_id))
            profile = envelope["data"]
        else:
            profile = kopyon_client.get_profile(str(trader_id))
            envelope = {"data": profile}
        params = {"traderId": str(trader_id)}
        raw_id = _save_raw(store, f"{_BASE}/traderProfile", params, envelope, started_at)
        store.upsert_history("traders", [dict(trader_id=str(trader_id), name=trader_name or profile.get("name"), source="kopyon",
                                               raw_response_id=raw_id, fetched_at=started_at, raw_fields=profile)])
        store.save_profile(str(trader_id), profile, raw_response_id=raw_id, fetched_at=started_at)
        record_success("traderProfile", end_ms)
        report.row_counts["profiles"] = 1
    except Exception as exc:
        record_error("traderProfile", exc)

    try:
        raw_method = getattr(kopyon_client, "get_chart_response", None)
        if raw_method:
            envelope = raw_method(str(trader_id), "ALL", "PNL")
            chart = envelope["data"]["points"]
        else:
            chart = kopyon_client.get_chart(str(trader_id), "ALL", "PNL")
            envelope = {"data": {"points": chart}}
        params = {"traderId": str(trader_id), "period": "ALL", "chartDataType": "PNL"}
        raw_id = _save_raw(store, f"{_BASE}/traderChart", params, envelope, started_at)
        store.save_chart_points(str(trader_id), "ALL", "PNL", chart, raw_response_id=raw_id, fetched_at=started_at)
        record_success("traderChart", end_ms)
        report.row_counts["charts"] = len(chart)
    except Exception as exc:
        record_error("traderChart", exc)

    # Current positions replace the previous set only after every page is visible and complete.
    resource = "openPositions"
    open_records, open_total, open_hidden = [], None, False
    try:
        page_method = getattr(kopyon_client, "iter_open_positions_pages", None)
        if page_method:
            pages = page_method(str(trader_id), market_type)
        else:
            data = kopyon_client.get_open_positions(str(trader_id), market_type)
            pages = [({"traderId": str(trader_id), "marketType": market_type}, data)]
        for params, response in pages:
            envelope, page = _unwrap_page(response)
            if not isinstance(page, dict) or not isinstance(page.get("positions"), list):
                raise ValueError("openPositions page is missing its positions list")
            raw_id = _save_raw(store, f"{_BASE}/openPositions", params, envelope, started_at)
            if page.get("hidden"):
                open_hidden = True
                break
            if page.get("total") is not None:
                open_total = int(page["total"])
            open_records.extend(normalize_position(str(trader_id), market_type, position, raw_response_id=raw_id, fetched_at=started_at)
                                for position in page["positions"])
        if open_hidden:
            raise RuntimeError("openPositions is hidden; retained the previous visible position set")
        if open_total is not None and len(open_records) < open_total:
            raise RuntimeError(f"openPositions incomplete: received {len(open_records)} of {open_total}")
        store.replace_open_positions(str(trader_id), market_type, open_records)
        record_success(resource, end_ms)
        report.row_counts["open_positions"] = len(open_records)
    except Exception as exc:
        record_error(resource, exc)

    for history_resource in ("positionHistory", "orderHistory"):
        prior = _checkpoint_watermark(store, str(trader_id), market_type, history_resource)
        if all_history:
            start_ms = None
        elif prior is not None:
            start_ms = max(0, int(prior) - overlap_ms)
        else:
            start_ms = max(0, end_ms - lookback_days * _DAY_MS)
        max_seen = prior or 0
        inserted = 0
        try:
            page_method = getattr(kopyon_client, "iter_history_pages", None)
            if page_method:
                pages = page_method(history_resource, str(trader_id), market_type, start_ms, end_ms)
            else:
                raise TypeError("Kopyon client must support iter_history_pages for auditable pagination")
            seen_cursors = set()
            for params, response in pages:
                envelope, page = _unwrap_page(response)
                if not isinstance(page, dict):
                    raise ValueError(f"{history_resource} page must be an object")
                raw_id = _save_raw(store, f"{_BASE}/{history_resource}", params, envelope, started_at)
                next_cursor = page.get("nextCursor")
                cursor = params.get("cursor")
                if next_cursor and (next_cursor == cursor or next_cursor in seen_cursors):
                    raise PaginationError(f"Repeated cursor in {history_resource}: {next_cursor}")
                if next_cursor:
                    seen_cursors.add(next_cursor)
                if page.get("hidden"):
                    raise RuntimeError(f"{history_resource} is hidden; sync remains incomplete")
                key = "positions" if history_resource == "positionHistory" else "orders"
                records = page.get(key)
                if not isinstance(records, list):
                    raise ValueError(f"{history_resource} page is missing {key} list")
                for source_record in records:
                    if history_resource == "positionHistory":
                        normalized = normalize_position(str(trader_id), market_type, source_record, raw_response_id=raw_id, fetched_at=started_at)
                        table = "position_history"
                    else:
                        normalized = normalize_order(str(trader_id), market_type, source_record, raw_response_id=raw_id, fetched_at=started_at)
                        table = "order_history"
                    store.upsert_history(table, [normalized])
                    inserted += 1
                    event_time = _timestamp(source_record, history_resource)
                    if event_time is not None:
                        max_seen = max(max_seen, event_time)
            # Advancing to the queried end also avoids re-reading a successfully checked empty interval.
            record_success(history_resource, max(max_seen, end_ms))
            report.row_counts["position_history" if history_resource == "positionHistory" else "order_history"] = inserted
        except Exception as exc:
            record_error(history_resource, exc)

    # Shared Binance market data uses one global checkpoint per symbol/interval,
    # so syncing several traders does not download identical public bars repeatedly.
    market_data_type = _BINANCE_MARKET_TYPE  # BinanceClient targets the documented USDⓈ-M public API.
    symbols = _market_symbols(store, str(trader_id), config.collection.market_type, config.collection.symbols)
    earliest_event = _earliest_trader_event(store, str(trader_id), config.collection.market_type)
    for symbol in symbols:
        for interval in config.collection.intervals:
            resource = f"klines:{symbol}:{interval}"
            checkpoint = store.get_checkpoint("__market_data__", market_data_type, resource)
            prior = checkpoint.get("watermark") if checkpoint else None
            coverage_start = _market_coverage_start(checkpoint)
            start_ms = _market_start_ms(prior, coverage_start, earliest_event, all_history, end_ms, lookback_days, overlap_ms)
            unfinished_start = _unfinished_kline_start(store, symbol, interval)
            if unfinished_start is not None:
                start_ms = min(start_ms, unfinished_start)
            inserted = 0
            try:
                pages = binance_client.iter_klines_pages(symbol, interval, start_ms, end_ms)
                for params, page in pages:
                    raw_id = _save_market_raw(store, "/fapi/v1/klines", params, page, started_at)
                    if not isinstance(page, list):
                        raise ValueError("Expected a Binance K-line page list")
                    page_times = [int(source[0]) for source in page]
                    _validate_market_page_times(page_times, int(params["startTime"]), end_ms, "klines")
                    rows = [_normalize_kline(symbol, interval, source, raw_id, started_at) for source in page]
                    store.upsert_history("klines", rows)
                    inserted += len(rows)
                next_coverage = _next_market_coverage_start(coverage_start, start_ms)
                record_market_success(resource, max(prior or 0, end_ms), next_coverage)
                report.row_counts[resource] = inserted
            except Exception as exc:
                previous = store.get_checkpoint("__market_data__", market_data_type, resource)
                store.save_checkpoint("__market_data__", market_data_type, resource,
                                      watermark=previous.get("watermark") if previous else None,
                                      cursor=previous.get("cursor") if previous else None,
                                      status="error", error=str(exc), updated_at=started_at)
                report.resources[resource] = "error"
                report.errors.append({"resource": resource, "error": str(exc)})

        resource = f"funding:{symbol}"
        checkpoint = store.get_checkpoint("__market_data__", market_data_type, resource)
        prior = checkpoint.get("watermark") if checkpoint else None
        coverage_start = _market_coverage_start(checkpoint)
        start_ms = _market_start_ms(prior, coverage_start, earliest_event, all_history, end_ms, lookback_days, overlap_ms)
        inserted = 0
        try:
            pages = binance_client.iter_funding_pages(symbol, start_ms, end_ms)
            for params, page in pages:
                raw_id = _save_market_raw(store, "/fapi/v1/fundingRate", params, page, started_at)
                if not isinstance(page, list):
                    raise ValueError("Expected a Binance funding-rate page list")
                page_times = [int(source["fundingTime"]) for source in page]
                _validate_market_page_times(page_times, int(params["startTime"]), end_ms, "fundingRate")
                rows = [_normalize_funding(symbol, source, raw_id, started_at) for source in page]
                store.upsert_history("funding_rates", rows)
                inserted += len(rows)
            next_coverage = _next_market_coverage_start(coverage_start, start_ms)
            record_market_success(resource, max(prior or 0, end_ms), next_coverage)
            report.row_counts[resource] = inserted
        except Exception as exc:
            previous = store.get_checkpoint("__market_data__", market_data_type, resource)
            store.save_checkpoint("__market_data__", market_data_type, resource,
                                  watermark=previous.get("watermark") if previous else None,
                                  cursor=previous.get("cursor") if previous else None,
                                  status="error", error=str(exc), updated_at=started_at)
            report.resources[resource] = "error"
            report.errors.append({"resource": resource, "error": str(exc)})

    report.finished_at = datetime.now(timezone.utc)
    report.complete = not report.errors
    return report
