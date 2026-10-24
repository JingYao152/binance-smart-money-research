import json
from datetime import datetime, timezone, timedelta

import pytest

from smartmoney_research.storage import ResearchStore
from smartmoney_research.normalize import normalize_position


@pytest.fixture
def store(tmp_path):
    with ResearchStore(tmp_path / 'test.duckdb') as db:
        db.initialize()
        yield db


def test_schema_and_reopen(tmp_path):
    path = tmp_path / 'test.duckdb'
    with ResearchStore(path) as db:
        db.initialize()
        db.initialize()
        tables = {row[0] for row in db.connection.execute('SHOW TABLES').fetchall()}
        assert {'schema_version', 'raw_responses', 'traders', 'profiles', 'charts', 'open_positions', 'position_history', 'order_history', 'klines', 'funding_rates', 'sync_state'} <= tables
    with ResearchStore(path) as db:
        assert db.connection.execute('SELECT version FROM schema_version').fetchone() == (1,)


@pytest.mark.parametrize('payload', [[['1', 2]], {'data': {'a': 1}}, None, 'scalar', True, 7])
def test_raw_snapshots_preserve_json_and_fetch_times(store, payload):
    now = datetime.now(timezone.utc)
    first = store.save_raw('binance', '/test', {'symbol': 'SYN'}, payload, 200, now)
    second = store.save_raw('binance', '/test', {'symbol': 'SYN'}, payload, 200, now + timedelta(seconds=1))
    rows = store.rows('raw_responses')
    assert first != second
    assert len(rows) == 2
    assert rows[0]['content_hash'] == rows[1]['content_hash']
    assert all(json.loads(row['payload']) == payload for row in rows)
    assert json.loads(rows[0]['params']) == {'symbol': 'SYN'}
    assert rows[0]['status'] == 200
    assert rows[0]['source'] == 'binance'
    assert rows[0]['endpoint'] == '/test'
    assert rows[0]['fetched_at'] != rows[1]['fetched_at']


def test_position_upsert_is_scoped_and_nulls_preserved(store):
    now = datetime.now(timezone.utc)
    raw = store.save_raw('kopyon', '/positionHistory', {'traderId': 'synthetic'}, {}, 200, now)
    row = normalize_position('synthetic', 'UM', {'positionId': 'p1', 'symbol': "SYN'USD", 'netPnl': '10'}, raw_response_id=raw, fetched_at=now)
    store.upsert_history('position_history', [row, row])
    row['net_pnl'] = 12
    store.upsert_history('position_history', [row])
    other = dict(row, trader_id='another')
    store.upsert_history('position_history', [other])
    rows = store.rows('position_history')
    assert len(rows) == 2
    assert rows[0]['net_pnl'] == 12
    assert rows[0]['trading_fee'] is None
    assert rows[0]['raw_response_id'] == raw


def test_raw_hash_is_canonical_and_tables_are_allowlisted(store):
    now = datetime.now(timezone.utc)
    for payload in ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}):
        store.save_raw('test', '/test', {}, payload, 200, now)
    assert len({r['content_hash'] for r in store.rows('raw_responses')}) == 1
    for table in ('raw_responses', 'position_history; DROP TABLE traders', 'unknown'):
        with pytest.raises(ValueError):
            store.upsert_history(table, [{}])
    with pytest.raises(ValueError):
        store.rows('traders; DROP TABLE traders')


def test_snapshot_and_checkpoint_interfaces(store):
    now = datetime.now(timezone.utc)
    raw = store.save_raw('test', '/test', {}, {}, 200, now)
    store.save_profile('synthetic', {'name': 'Synthetic'}, raw_response_id=raw, fetched_at=now)
    store.save_profile('synthetic', {'name': 'Synthetic'}, raw_response_id=raw, fetched_at=now)
    assert len(store.rows('profiles')) == 2
    store.save_chart_points('synthetic', '1Y', 'PNL', [{'timestamp': 1000, 'value': '2'}], raw_response_id=raw, fetched_at=now)
    assert store.rows('charts')[0]['value'] == 2
    pos = normalize_position('synthetic', 'UM', {'positionId': 'p'}, raw_response_id=raw, fetched_at=now)
    store.replace_open_positions('synthetic', 'UM', [pos])
    assert len(store.rows('open_positions')) == 1
    with pytest.raises(ValueError):
        store.replace_open_positions('synthetic', 'UM', [dict(pos, trader_id='other')])
    assert len(store.rows('open_positions')) == 1
    store.replace_open_positions('synthetic', 'UM', [])
    assert store.rows('open_positions') == []
    store.save_checkpoint('synthetic', 'UM', 'positionHistory', watermark=1000, status='complete')
    store.save_checkpoint('synthetic', 'UM', 'positionHistory', watermark=1000, status='error', error='cursor cycle')
    checkpoint = store.get_checkpoint('synthetic', 'UM', 'positionHistory')
    assert checkpoint['watermark'] == 1000
    assert checkpoint['error'] == 'cursor cycle'
    assert store.get_checkpoint('absent', 'UM', 'positionHistory') is None


def test_failed_upsert_rolls_back_the_whole_batch(store):
    valid = normalize_position('synthetic', 'UM', {'positionId': 'good', 'grossPnl': '3'})
    invalid = normalize_position('synthetic', 'UM', {'positionId': 'bad', 'grossPnl': 'invalid'})
    with pytest.raises(Exception):
        store.upsert_history('position_history', [valid, invalid])
    assert store.rows('position_history') == []
