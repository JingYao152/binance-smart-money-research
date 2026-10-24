"""Offline dashboard boundaries use invented records only."""
from datetime import datetime, timezone
import hashlib
import socket

import duckdb
import pytest

from smartmoney_research.dashboard import load_dashboard_data, render_dashboard
from smartmoney_research.storage import ResearchStore


def test_missing_database_is_empty_and_does_not_create_file(tmp_path):
    path = tmp_path / 'absent.duckdb'
    data = load_dashboard_data(path)
    assert data['status'] == 'missing_database'
    assert data['traders'] == []
    assert not path.exists()


def test_empty_dashboard_is_local_read_only_and_renders(tmp_path, monkeypatch):
    path = tmp_path / 'empty.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('network forbidden'))
    original = duckdb.connect
    calls = []
    def connect(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(duckdb, 'connect', connect)
    data = load_dashboard_data(path)
    assert data['traders'] == []
    class UI:
        def title(self, *args): pass
        def caption(self, *args): pass
        def info(self, *args): pass
    render_dashboard(UI(), data)
    assert calls and all(call.get('read_only') is True for call in calls)
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()


def test_dashboard_exposes_hidden_history_and_missing_fields(tmp_path):
    path = tmp_path / 'research.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.save_raw('kopyon', '/v5/binance-smart-money/positionHistory',
                       {'traderId': 'synthetic'}, {'data': {'hidden': True}}, 200,
                       datetime.now(timezone.utc))
        store.upsert_history('position_history', [dict(trader_id='synthetic', market_type='UM',
                             record_id='fixture', symbol='ETHUSDT', open_time=1000)])
    data = load_dashboard_data(path, trader_id='synthetic')
    assert data['coverage']['hidden_history']
    assert data['coverage']['missing_fields']['position_history']['net_pnl'] == 1
    assert data['analysis']['trade_events'][0]['feature_status'] == 'missing_market_bars'
    assert data['profiles'] == []


def test_ready_dashboard_render_has_no_source_calls_and_handles_null_chart_samples(tmp_path, monkeypatch):
    path = tmp_path / 'ready.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.save_profile('synthetic', {'sharing': True})
        store.save_chart_points('synthetic', '1Y', 'PNL', [{'timestamp': 1000, 'value': None}])
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('network forbidden'))
    data = load_dashboard_data(path)
    calls = []
    class UI:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                calls.append(name)
            return record
    render_dashboard(UI(), data)
    assert 'dataframe' in calls and 'json' in calls


def test_event_before_first_close_preserves_missing_feature_status(tmp_path):
    path = tmp_path / 'early.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.upsert_history('position_history', [dict(trader_id='synthetic', market_type='UM',
                             record_id='fixture', symbol='ETHUSDT', open_time=1999, close_time=2000)])
        store.upsert_history('klines', [dict(symbol='ETHUSDT', interval='1h', open_time=1000,
                             close_time=1999, fetched_at=datetime.fromtimestamp(2, timezone.utc),
                             open=100, high=101, low=99, close=100, volume=10)])
    events = load_dashboard_data(path)['analysis']['trade_events']
    assert events[0]['feature_status'] == 'no_prior_closed_bar'
    assert events[0]['bar_close_time'] is None
    assert events[1]['feature_status'] == 'aligned'
    assert events[1]['bar_close_time'] == 1999
    assert __import__('pandas').isna(events[1]['rsi'])


def test_bar_retrieved_before_it_closed_cannot_supply_event_features(tmp_path):
    from smartmoney_research.dashboard import analyze_local
    path = tmp_path / 'partial-bar.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.upsert_history('position_history', [dict(trader_id='synthetic', market_type='UM',
                             record_id='fixture', symbol='ETHUSDT', open_time=4000)])
        store.upsert_history('klines', [dict(symbol='ETHUSDT', interval='1h', open_time=0,
                             close_time=999, open=100, high=101, low=99, close=100, volume=10,
                             fetched_at=datetime.fromtimestamp(1, timezone.utc)),
                             dict(symbol='ETHUSDT', interval='1h', open_time=1000,
                             close_time=1999, open=200, high=201, low=199, close=200, volume=10,
                             fetched_at=datetime.fromtimestamp(1.5, timezone.utc))])
        result = analyze_local(store, 'synthetic')
    assert result['trade_events'][0]['bar_close_time'] == 999


def test_report_comparisons_skip_other_traders():
    seen = []
    class UI:
        def __getattr__(self, name):
            def record(*args, **kwargs):
                seen.append((name, args))
            return record
    run = dict(single=dict(metrics={'total_return': .1}, assumptions={'leverage': 1},
                           benchmark={'available': False}, funding={}, liquidation_risk={}),
               walk_forward=[], sensitivity={})
    data = dict(status='ready', trader_id='synthetic', profiles=[], coverage={}, positions=[],
                charts=[], analysis={'trade_events': []})
    reports = [dict(trader_id='other-fixture', symbols={'WRONG': run}),
               dict(trader_id='synthetic', rule={'kind': 'ema_crossover'}, symbols={'ETHUSDT': run})]
    render_dashboard(UI(), data, reports)
    json_symbols = [args[0]['symbol'] for name, args in seen if name == 'json' and 'symbol' in args[0]]
    assert json_symbols == ['ETHUSDT']
    assert any(name == 'info' and 'another trader' in args[0] for name, args in seen)


def test_malformed_uploaded_reports_do_not_abort_valid_comparisons():
    seen = []
    class UI:
        def __getattr__(self, name):
            return lambda *args, **kwargs: seen.append((name, args))
    data = dict(status='ready', trader_id='synthetic', profiles=[], coverage={}, positions=[],
                charts=[], analysis={'trade_events': []})
    valid = dict(single=dict(metrics={'total_return': .1}, assumptions={}, benchmark={}, funding={},
                             liquidation_risk={}), walk_forward=[], sensitivity={})
    reports = [None, dict(trader_id='synthetic', symbols={'BAD': dict(single=dict(metrics={}))}),
               dict(trader_id='synthetic', symbols={'WRONG_TYPE': []}),
               dict(trader_id='synthetic', rule=[], symbols={'BAD_RULE': valid}),
               dict(trader_id='synthetic', symbols={'GOOD': valid})]
    render_dashboard(UI(), data, reports)
    assert [args[0]['symbol'] for name, args in seen if name == 'json' and 'symbol' in args[0]] == ['GOOD']
    assert sum(name == 'error' for name, args in seen) == 4


def test_market_checkpoint_identity_is_not_a_dashboard_trader(tmp_path):
    path = tmp_path / 'market-checkpoint.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.save_checkpoint('__market_data__', 'UM', 'klines:BTCUSDT:1h', watermark=1000, status='complete')
        store.save_profile('synthetic', {'sharing': True})
    data = load_dashboard_data(path)
    assert data['traders'] == ['synthetic']
    assert data['trader_id'] == 'synthetic'


@pytest.mark.parametrize('metadata,future_close', [('missing_retrieval', False),
                                                       ('missing_retrieval', True), ('future_close', True)])
def test_unverified_bars_are_unavailable_for_analytics(tmp_path, metadata, future_close):
    from datetime import timedelta
    from smartmoney_research.dashboard import local_bars
    now = datetime.now(timezone.utc)
    future_ms = int((now + timedelta(hours=1)).timestamp() * 1000)
    close_ms = future_ms if future_close else 1999
    fetched_at = None if metadata == 'missing_retrieval' else now + timedelta(hours=2)
    path = tmp_path / 'unverified.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        store.upsert_history('position_history', [dict(trader_id='synthetic', market_type='UM',
                             record_id='fixture', symbol='ETHUSDT', open_time=future_ms + 1)])
        store.upsert_history('klines', [dict(symbol='ETHUSDT', interval='1h', open_time=1000,
                             close_time=close_ms, fetched_at=fetched_at,
                             open=100, high=101, low=99, close=100, volume=10)])
        assert local_bars(store, 'ETHUSDT', '1h').empty
    data = load_dashboard_data(path)
    assert data['analysis']['trade_events'][0]['feature_status'] == 'unavailable_market_bars'
    coverage = data['coverage']['market_bars']['ETHUSDT']
    assert coverage['stored_rows'] == 1
    assert coverage['verified_rows'] == 0
    assert coverage['unavailable_rows'] == 1
    assert coverage[metadata] == 1
    assert coverage['future_close'] == int(future_close)
