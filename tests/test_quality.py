from datetime import datetime, timezone

from smartmoney_research.storage import ResearchStore
from smartmoney_research.normalize import normalize_position, normalize_order
from smartmoney_research.quality import check_integrity


def test_hidden_missing_timestamps_residuals_and_scope(tmp_path):
    with ResearchStore(tmp_path / 'quality.duckdb') as store:
        store.initialize()
        now = datetime.now(timezone.utc)
        raw = store.save_raw('kopyon', '/positionHistory', {'traderId': 'synthetic', 'marketType': 'UM'}, {'data': {'hidden': True, 'positions': []}}, 200, now)
        store.save_raw('kopyon', '/orderHistory', {'traderId': 'other'}, {'data': {'hidden': True}}, 200, now)
        records = [
            {'positionId': 'mismatch', 'symbol': 'SYN', 'openedAt': 2000, 'closedAt': 1000, 'grossPnl': '10', 'fundingPnl': '-2', 'tradingFee': '1', 'liquidationFee': '0', 'netPnl': '9'},
            {'positionId': 'missing', 'symbol': 'SYN', 'openedAt': 1000, 'netPnl': '9'},
        ]
        store.upsert_history('position_history', [normalize_position('synthetic', 'UM', r, raw_response_id=raw, fetched_at=now) for r in records])
        store.upsert_history('order_history', [normalize_order('synthetic', 'UM', {'symbol': 'SYN', 'createdAt': -1})])
        before = store.rows('position_history')
        report = check_integrity(store, 'synthetic')
        assert report['incomplete'] is True
        assert len(report['hidden_history']) == 1
        assert report['counts']['position_history'] == 2
        assert report['missing_fields']['position_history']['trading_fee'] == 1
        assert report['duplicate_counts']['position_history'] == 0
        assert len(report['timestamp_errors']) == 2
        assert report['pnl_residuals'] == [{'record_id': 'mismatch', 'market_type': 'UM', 'residual': 2.0}]
        assert report['pnl_unverifiable_count'] == 1
        assert store.rows('position_history') == before


def test_cursor_cycles_are_scoped_to_resource_market_and_run(tmp_path):
    with ResearchStore(tmp_path / 'cursor.duckdb') as store:
        store.initialize()
        now = datetime.now(timezone.utc)
        for endpoint, market, cursor, next_cursor in [('/positionHistory', 'UM', None, 'c1'), ('/positionHistory', 'UM', 'c1', 'c1'), ('/orderHistory', 'UM', None, 'c1'), ('/positionHistory', 'CM', None, 'c1')]:
            params = {'traderId': 'synthetic', 'marketType': market}
            if cursor:
                params['cursor'] = cursor
            store.save_raw('kopyon', endpoint, params, {'data': {'nextCursor': next_cursor}}, 200, now)
        report = check_integrity(store, 'synthetic')
        assert len(report['cursor_errors']) == 1
        assert report['incomplete'] is True


def test_empty_and_failed_sync_are_distinct_from_complete_coverage(tmp_path):
    with ResearchStore(tmp_path / 'empty.duckdb') as store:
        store.initialize()
        empty = check_integrity(store, 'synthetic')
        assert empty['incomplete'] is True
        assert empty['coverage_status'] == 'uncollected'
        store.save_checkpoint('synthetic', 'UM', 'positionHistory', status='error', error='HTTP 503')
        failed = check_integrity(store, 'synthetic')
        assert failed['sync_errors'][0]['error'] == 'HTTP 503'
        assert failed['cursor_errors'] == []


def test_successful_empty_response_is_observed_without_inventing_rows(tmp_path):
    with ResearchStore(tmp_path / 'observed.duckdb') as store:
        store.initialize()
        store.save_raw('kopyon', '/positionHistory', {'traderId': 'synthetic'}, {'data': {'positions': [], 'hidden': False}}, 200, datetime.now(timezone.utc))
        report = check_integrity(store, 'synthetic')
        assert report['coverage_status'] == 'observed'
        assert report['counts']['position_history'] == 0
        assert report['incomplete'] is False


def test_updated_at_is_checked_for_positions_and_orders(tmp_path):
    with ResearchStore(tmp_path / 'updated.duckdb') as store:
        store.initialize()
        fetched_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        future = int(fetched_at.timestamp() * 1000) + 120_000
        store.upsert_history('position_history', [
            normalize_position('synthetic', 'UM', {'positionId': 'negative', 'updatedAt': -1}, fetched_at=fetched_at),
        ])
        store.upsert_history('open_positions', [
            normalize_position('synthetic', 'UM', {'positionId': 'future', 'updatedAt': future}, fetched_at=fetched_at),
        ])
        store.upsert_history('order_history', [
            normalize_order('synthetic', 'UM', {'updatedAt': value}, fetched_at=fetched_at)
            for value in (-1, future)
        ])
        errors = check_integrity(store, 'synthetic')['timestamp_errors']
        assert len(errors) == 4
        assert {error['table'] for error in errors} == {'position_history', 'open_positions', 'order_history'}
        messages = [message for error in errors for message in error['errors']]
        assert messages.count('updated_at: negative timestamp') == 2
        assert messages.count('updated_at: after retrieval') == 2
