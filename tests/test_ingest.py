from datetime import datetime, timedelta, timezone
import json

import pytest

from smartmoney_research.clients.http import PaginationError
from smartmoney_research.config import CollectionSettings, Settings, StorageSettings
from smartmoney_research.ingest import sync_trader
from smartmoney_research.storage import ResearchStore


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)


class FakeKopyon:
    def __init__(self, history_pages=None, open_pages=None):
        self.history_pages = history_pages or {}
        self.open_pages = open_pages or {}
        self.calls = []

    def get_profile(self, trader_id):
        self.calls.append(("profile", trader_id))
        return {"traderId": trader_id, "name": "Synthetic Trader"}

    def get_chart(self, trader_id, period, chart_type):
        self.calls.append(("chart", trader_id, period, chart_type))
        return [{"timestamp": NOW_MS, "value": 10}]

    def iter_open_positions_pages(self, trader_id, market_type):
        self.calls.append(("open_positions", trader_id, market_type))
        pages = self.open_pages.get(market_type, [{"positions": [], "total": 0, "hidden": False}])
        for index, page in enumerate(pages, 1):
            yield ({"traderId": trader_id, "marketType": market_type, "page": index, "rows": 20}, page)

    def iter_history_pages(self, resource, trader_id, market_type, start_ms, end_ms, rows=20):
        self.calls.append((resource, trader_id, market_type, start_ms, end_ms))
        pages = self.history_pages.get(resource, [{"positions": [], "orders": []}])
        cursor = None
        for page in pages:
            params = {"traderId": trader_id, "marketType": market_type, "rows": rows}
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
            if cursor is not None:
                params["cursor"] = cursor
            yield params, page
            cursor = page.get("nextCursor")
            if cursor is None:
                break


@pytest.fixture
def store(tmp_path):
    with ResearchStore(tmp_path / "sync.duckdb") as db:
        db.initialize()
        yield db


def settings(tmp_path, *, overlap_ms=1000, days=365, symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"), intervals=("1h",)):
    return Settings(collection=CollectionSettings(days=days, market_type="UM", overlap_ms=overlap_ms, symbols=symbols, intervals=intervals), storage=StorageSettings(path=str(tmp_path / "sync.duckdb")))


class FakeBinance:
    def __init__(self, klines=None, funding=None):
        self.klines = klines or {}
        self.funding = funding or {}
        self.calls = []

    def iter_klines_pages(self, symbol, interval, start_ms, end_ms):
        self.calls.append(("klines", symbol, interval, start_ms, end_ms))
        payload = self.klines.get((symbol, interval), [])
        yield ({"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1500}, payload)

    def iter_funding_pages(self, symbol, start_ms, end_ms):
        self.calls.append(("funding", symbol, start_ms, end_ms))
        payload = self.funding.get(symbol, [])
        yield ({"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000}, payload)


def test_default_history_window_is_365_days_and_raw_pages_are_saved_once(store, tmp_path):
    pages = {
        "positionHistory": [
            {"positions": [{"positionId": "pos-a", "symbol": "SYNUSDT", "openedAt": NOW_MS - 10}], "nextCursor": "cursor-a"},
            {"positions": [{"positionId": "pos-b", "symbol": "SYNUSDT", "openedAt": NOW_MS - 5}]},
        ],
        "orderHistory": [{"orders": [{"symbol": "SYNUSDT", "side": "BUY", "createdAt": NOW_MS - 4, "quantity": "1"}]}],
    }
    api = FakeKopyon(history_pages=pages)
    report = sync_trader(settings(tmp_path), {"trader_id": "synthetic-id", "name": "Synthetic Trader"}, store, kopyon_client=api, binance_client=FakeBinance(), now=NOW)

    expected_start = NOW_MS - 365 * 24 * 60 * 60 * 1000
    assert report.complete is True
    assert [call[3] for call in api.calls if call[0] in ("positionHistory", "orderHistory")] == [expected_start, expected_start]
    assert len(store.rows("position_history")) == 2
    assert len(store.rows("order_history")) == 1
    raw_pages = [row for row in store.rows("raw_responses") if "positionHistory" in row["endpoint"]]
    assert len(raw_pages) == 2
    assert all(row["raw_response_id"] for row in store.rows("position_history"))


def test_all_history_omits_lower_bound(store, tmp_path):
    api = FakeKopyon()
    report = sync_trader(settings(tmp_path), {"trader_id": "synthetic-id"}, store, all_history=True, kopyon_client=api, binance_client=FakeBinance(), now=NOW)
    assert report.complete is True
    assert all(call[3] is None for call in api.calls if call[0] in ("positionHistory", "orderHistory"))


def test_repeat_sync_is_idempotent_and_uses_checkpoint_overlap(store, tmp_path):
    pages = {"positionHistory": [{"positions": [{"positionId": "pos-a", "symbol": "SYNUSDT", "openedAt": NOW_MS - 10}]}]}
    api = FakeKopyon(history_pages=pages)
    cfg = settings(tmp_path, overlap_ms=500)
    trader = {"trader_id": "synthetic-id"}
    first = sync_trader(cfg, trader, store, kopyon_client=api, binance_client=FakeBinance(), now=NOW)
    raw_count = len(store.rows("raw_responses"))
    second = sync_trader(cfg, trader, store, kopyon_client=api, binance_client=FakeBinance(), now=NOW, lookback_days=30)
    position_calls = [call for call in api.calls if call[0] == "positionHistory"]

    assert first.complete and second.complete
    assert len(store.rows("position_history")) == 1
    assert len(store.rows("order_history")) == 0
    assert len(store.rows("raw_responses")) > raw_count
    # The checkpoint records the latest fully scanned event time, including an empty tail interval.
    assert position_calls[1][3] == NOW_MS - 500


def test_cursor_cycle_fails_without_advancing_checkpoint(store, tmp_path):
    class Cycling(FakeKopyon):
        def iter_history_pages(self, resource, trader_id, market_type, start_ms, end_ms, rows=20):
            self.calls.append((resource, trader_id, market_type, start_ms, end_ms))
            page = {"positions": [{"positionId": "pos-a", "openedAt": NOW_MS - 10}], "nextCursor": "repeat"}
            yield ({"traderId": trader_id, "marketType": market_type}, page)
            yield ({"traderId": trader_id, "marketType": market_type, "cursor": "repeat"}, page)

    store.save_checkpoint("synthetic-id", "UM", "positionHistory", watermark=NOW_MS - 1000, status="complete")
    report = sync_trader(settings(tmp_path), {"trader_id": "synthetic-id"}, store, kopyon_client=Cycling(), binance_client=FakeBinance(), now=NOW)

    state = store.get_checkpoint("synthetic-id", "UM", "positionHistory")
    assert report.complete is False
    assert any("cursor" in error["error"].lower() for error in report.errors)
    assert state["status"] == "error"
    assert state["watermark"] == NOW_MS - 1000


def test_hidden_open_positions_are_not_used_to_delete_existing_rows(store, tmp_path):
    from smartmoney_research.normalize import normalize_position

    existing = normalize_position("synthetic-id", "UM", {"positionId": "keep-me", "symbol": "SYNUSDT"})
    store.replace_open_positions("synthetic-id", "UM", [existing])
    api = FakeKopyon(open_pages={"UM": [{"positions": [], "total": 0, "hidden": True}]})
    report = sync_trader(settings(tmp_path), {"trader_id": "synthetic-id"}, store, kopyon_client=api, binance_client=FakeBinance(), now=NOW)
    state = store.get_checkpoint("synthetic-id", "UM", "openPositions")

    assert report.complete is False
    assert [row["record_id"] for row in store.rows("open_positions")] == ["keep-me"]
    assert state["status"] == "error"


def test_market_sync_saves_exact_binance_pages_and_includes_observed_symbols(store, tmp_path):
    raw_kline = [NOW_MS - 3_600_000, "10", "11", "9", "10.5", "100", NOW_MS - 1, "1050", 2, "40", "420", "0"]
    raw_funding = {"symbol": "BTCUSDT", "fundingTime": NOW_MS - 1000, "fundingRate": "0.0001", "markPrice": "10.5"}
    kopyon = FakeKopyon(history_pages={"positionHistory": [{"positions": [{"positionId": "sol-pos", "symbol": "SOLUSDT", "openedAt": NOW_MS - 10}]}]})
    market = FakeBinance(klines={("BTCUSDT", "1h"): [raw_kline]}, funding={"BTCUSDT": [raw_funding]})

    report = sync_trader(settings(tmp_path, symbols=("BTCUSDT",)), {"trader_id": "synthetic-id"}, store,
                         kopyon_client=kopyon, binance_client=market, now=NOW)

    assert report.complete is True
    assert {call[1] for call in market.calls} == {"BTCUSDT", "SOLUSDT"}
    assert len(store.rows("klines")) == 1
    assert store.rows("klines")[0]["raw_response_id"]
    assert json.loads(store.rows("klines")[0]["raw_fields"]) == raw_kline
    assert len(store.rows("funding_rates")) == 1
    assert store.rows("funding_rates")[0]["raw_response_id"]
    raw_binance = [row for row in store.rows("raw_responses") if row["source"] == "binance"]
    assert len(raw_binance) == 4
    assert any(json.loads(row["payload"]) == [raw_kline] for row in raw_binance)
    assert any(json.loads(row["payload"]) == [raw_funding] for row in raw_binance)
    assert store.get_checkpoint("__market_data__", "UM", "klines:BTCUSDT:1h")["status"] == "complete"
    assert store.get_checkpoint("__market_data__", "UM", "funding:BTCUSDT")["status"] == "complete"


def test_market_sync_uses_configured_lookback_and_overlap(store, tmp_path):
    cfg = settings(tmp_path, overlap_ms=2000, days=42, symbols=("BTCUSDT",))
    market = FakeBinance()
    trader = {"trader_id": "synthetic-id"}
    first = sync_trader(cfg, trader, store, lookback_days=42, kopyon_client=FakeKopyon(), binance_client=market, now=NOW)
    second = sync_trader(cfg, trader, store, lookback_days=42, kopyon_client=FakeKopyon(), binance_client=market, now=NOW)
    kline_calls = [call for call in market.calls if call[0] == "klines"]

    assert first.complete and second.complete
    assert kline_calls[0][3] == NOW_MS - 42 * 24 * 60 * 60 * 1000
    assert kline_calls[1][3] == NOW_MS - 2000


def test_market_all_history_starts_at_earliest_trade_event(store, tmp_path):
    market = FakeBinance()
    earliest = NOW_MS - 123_456_789
    kopyon = FakeKopyon(history_pages={"positionHistory": [{"positions": [{"positionId": "old-position", "openedAt": earliest}]}]})
    cfg = settings(tmp_path, symbols=("BTCUSDT",), overlap_ms=1000)
    first = sync_trader(cfg, {"trader_id": "synthetic-id"}, store,
                        kopyon_client=kopyon, binance_client=market, all_history=True, now=NOW)
    second = sync_trader(cfg, {"trader_id": "synthetic-id"}, store,
                         kopyon_client=kopyon, binance_client=market, all_history=True, now=NOW)
    kline_calls = [call for call in market.calls if call[0] == "klines"]
    funding_calls = [call for call in market.calls if call[0] == "funding"]
    assert first.complete and second.complete
    assert [call[3] for call in kline_calls] == [earliest, NOW_MS - 1000]
    assert [call[2] for call in funding_calls] == [earliest, NOW_MS - 1000]


def test_all_history_upgrades_old_recent_window_checkpoint_once(store, tmp_path):
    earliest = NOW_MS - 123_456_789
    recent_watermark = NOW_MS - 10_000
    store.save_checkpoint("__market_data__", "UM", "klines:BTCUSDT:1h", watermark=recent_watermark, status="complete")
    store.save_checkpoint("__market_data__", "UM", "funding:BTCUSDT", watermark=recent_watermark, status="complete")
    kopyon = FakeKopyon(history_pages={"positionHistory": [{"positions": [{"positionId": "old-position", "openedAt": earliest}]}]})
    market = FakeBinance()
    cfg = settings(tmp_path, symbols=("BTCUSDT",), overlap_ms=1000)

    first = sync_trader(cfg, {"trader_id": "synthetic-id"}, store, kopyon_client=kopyon, binance_client=market, all_history=True, now=NOW)
    first_start = [call[3] for call in market.calls if call[0] == "klines"][0]
    first_checkpoint = store.get_checkpoint("__market_data__", "UM", "klines:BTCUSDT:1h")
    market.calls.clear()
    second = sync_trader(cfg, {"trader_id": "synthetic-id"}, store, kopyon_client=kopyon, binance_client=market, all_history=True, now=NOW)
    second_start = [call[3] for call in market.calls if call[0] == "klines"][0]

    assert first.complete and second.complete
    assert first_start == earliest
    assert first_checkpoint["cursor"] == f"market_coverage_start_ms={earliest}"
    assert second_start == NOW_MS - 1000


def test_market_all_history_without_event_times_uses_configured_window(store, tmp_path):
    market = FakeBinance()
    cfg = settings(tmp_path, days=42, symbols=("BTCUSDT",))
    report = sync_trader(cfg, {"trader_id": "synthetic-id"}, store, lookback_days=42, all_history=True,
                         kopyon_client=FakeKopyon(), binance_client=market, now=NOW)
    assert report.complete
    assert all(call[3] == NOW_MS - 42 * 24 * 60 * 60 * 1000 for call in market.calls if call[0] == "klines")
    assert all(call[2] == NOW_MS - 42 * 24 * 60 * 60 * 1000 for call in market.calls if call[0] == "funding")


def test_market_failure_keeps_previous_symbol_interval_watermark(store, tmp_path):
    class BrokenKlines(FakeBinance):
        def iter_klines_pages(self, symbol, interval, start_ms, end_ms):
            raise RuntimeError("synthetic market failure")
            yield

    previous = NOW_MS - 10_000
    coverage = NOW_MS - 42 * 24 * 60 * 60 * 1000
    store.save_checkpoint("__market_data__", "UM", "klines:BTCUSDT:1h", watermark=previous,
                          cursor=f"market_coverage_start_ms={coverage}", status="complete")
    report = sync_trader(settings(tmp_path, symbols=("BTCUSDT",)), {"trader_id": "synthetic-id"}, store,
                         kopyon_client=FakeKopyon(), binance_client=BrokenKlines(), now=NOW)
    checkpoint = store.get_checkpoint("__market_data__", "UM", "klines:BTCUSDT:1h")

    assert report.complete is False
    assert checkpoint["status"] == "error"
    assert checkpoint["watermark"] == previous
    assert checkpoint["cursor"] == f"market_coverage_start_ms={coverage}"
    assert any(error["resource"] == "klines:BTCUSDT:1h" for error in report.errors)


@pytest.mark.parametrize('interval,duration_days,overlap_ms', [
    ('1h', 1 / 24, 0), ('1w', 7, 86_400_000), ('1M', 31, 86_400_000),
])
def test_unfinished_klines_are_refetched_after_close(store, tmp_path, interval, duration_days, overlap_ms):
    from smartmoney_research.dashboard import local_bars
    opened = NOW - timedelta(days=duration_days / 2)
    closed = opened + timedelta(days=duration_days)
    open_ms = int(opened.timestamp() * 1000)
    close_ms = int(closed.timestamp() * 1000) - 1
    candle = [open_ms, '10', '11', '9', '10', '100', close_ms, '1000', 2, '40', '400', '0']

    class PagingMarket(FakeBinance):
        def iter_klines_pages(self, symbol, interval, start_ms, end_ms):
            self.calls.append(('klines', symbol, interval, start_ms, end_ms))
            payload = [candle.copy()] if start_ms <= open_ms <= end_ms else []
            if payload and end_ms > close_ms:
                payload[0][4] = '10.5'
            yield dict(symbol=symbol, interval=interval, startTime=start_ms, endTime=end_ms), payload

    market = PagingMarket()
    cfg = settings(tmp_path, symbols=('BTCUSDT',), intervals=(interval,), overlap_ms=overlap_ms)
    first = sync_trader(cfg, 'synthetic-id', store, kopyon_client=FakeKopyon(), binance_client=market, now=NOW)
    assert first.complete
    assert local_bars(store, 'BTCUSDT', interval).empty
    second_time = closed + timedelta(hours=1)
    second = sync_trader(cfg, 'synthetic-id', store, kopyon_client=FakeKopyon(), binance_client=market, now=second_time)
    assert second.complete
    bars = local_bars(store, 'BTCUSDT', interval)
    assert bars['open_time'].tolist() == [open_ms]
    assert bars.iloc[0]['close'] == 10.5
    assert market.calls[-2][3] == open_ms
    assert store.get_checkpoint('__market_data__', 'UM', f'klines:BTCUSDT:{interval}')['watermark'] == int(second_time.timestamp() * 1000)


def test_empty_market_ranges_advance_checkpoints_with_zero_overlap(store, tmp_path):
    market = FakeBinance()
    cfg = settings(tmp_path, symbols=('BTCUSDT',), overlap_ms=0)
    for now in (NOW, NOW + timedelta(hours=1)):
        assert sync_trader(cfg, 'synthetic-id', store, kopyon_client=FakeKopyon(), binance_client=market, now=now).complete
    kline_calls = [call for call in market.calls if call[0] == 'klines']
    assert kline_calls[1][3] == NOW_MS
    assert store.get_checkpoint('__market_data__', 'UM', 'klines:BTCUSDT:1h')['watermark'] == NOW_MS + 3_600_000
