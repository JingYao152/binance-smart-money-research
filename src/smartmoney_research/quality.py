"""Read-only coverage diagnostics. Never infer missing costs or repair source PnL."""
from collections import Counter
from decimal import Decimal
import json

_CRITICAL = {
    'position_history': ('symbol', 'side', 'open_time', 'gross_pnl', 'funding_pnl', 'trading_fee', 'liquidation_fee', 'net_pnl', 'raw_response_id', 'fetched_at'),
    'open_positions': ('symbol', 'side', 'entry_price', 'size', 'raw_response_id', 'fetched_at'),
    'order_history': ('symbol', 'side', 'timestamp', 'price', 'quantity', 'raw_response_id', 'fetched_at'),
}


def _hidden_paths(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f'{path}.{key}' if path else key
            if key == 'hidden' and item is True:
                yield child
            elif key.lower().startswith('show') and item is False:
                yield child
            yield from _hidden_paths(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _hidden_paths(item, f'{path}[{index}]')


def check_integrity(store, trader_id: str):
    """Scope raw cursor streams by source, route, market and time-window parameters.

    A request without cursor starts a fresh stream. Repeated returned cursors or
    non-advancing request/response cursors are errors. Missing PnL components yield
    an unverifiable count rather than a residual calculated with invented zeros.
    """
    report = dict(counts={}, hidden_history=[], missing_fields={}, duplicate_counts={}, cursor_errors=[], timestamp_errors=[], pnl_residuals=[], pnl_unverifiable_count=0, fingerprint_counts={}, sync_errors=[])
    streams = {}
    successful_responses = 0
    for raw in store.rows('raw_responses'):
        params = json.loads(raw['params'])
        if params.get('traderId', params.get('trader_id')) != trader_id:
            continue
        successful_responses += 200 <= raw['status'] < 300
        payload = json.loads(raw['payload'])
        for path in _hidden_paths(payload):
            report['hidden_history'].append(dict(raw_response_id=raw['raw_response_id'], endpoint=raw['endpoint'], field=path))
        if not isinstance(payload, dict):
            continue
        data = payload.get('data', payload)
        if not isinstance(data, dict):
            continue
        cursor, next_cursor = params.get('cursor'), data.get('nextCursor')
        scope = (raw['source'], raw['endpoint'], json.dumps({k: v for k, v in params.items() if k != 'cursor'}, sort_keys=True))
        if not cursor:
            streams[scope] = set()
        seen = streams.setdefault(scope, set())
        if next_cursor and (next_cursor == cursor or next_cursor in seen):
            report['cursor_errors'].append(dict(raw_response_id=raw['raw_response_id'], endpoint=raw['endpoint'], cursor=next_cursor))
        if next_cursor:
            seen.add(next_cursor)
    for table, fields in _CRITICAL.items():
        rows = [r for r in store.rows(table) if r['trader_id'] == trader_id]
        report['counts'][table] = len(rows)
        report['missing_fields'][table] = {field: sum(r[field] is None for r in rows) for field in fields}
        counts = Counter((r['market_type'], r['record_id']) for r in rows)
        report['duplicate_counts'][table] = sum(count - 1 for count in counts.values())
        report['fingerprint_counts'][table] = sum(r['id_kind'] == 'content_sha256' for r in rows)
        for row in rows:
            times = ('open_time', 'close_time', 'updated_at') if table != 'order_history' else ('timestamp', 'updated_at')
            errors = []
            for field in times:
                value = row[field]
                if value is not None and value < 0:
                    errors.append(f'{field}: negative timestamp')
                if value is not None and row['fetched_at'] and value > row['fetched_at'].timestamp() * 1000 + 60_000:
                    errors.append(f'{field}: after retrieval')
            if table != 'order_history' and row['open_time'] is not None and row['close_time'] is not None and row['close_time'] < row['open_time']:
                errors.append('close_time precedes open_time')
            if errors:
                report['timestamp_errors'].append(dict(table=table, record_id=row['record_id'], errors=errors))
            if table == 'position_history':
                components = ('net_pnl', 'gross_pnl', 'funding_pnl', 'trading_fee', 'liquidation_fee')
                if any(row[field] is None for field in components):
                    report['pnl_unverifiable_count'] += 1
                    continue
                net, gross, funding, fee, liquidation_fee = [Decimal(str(row[field])) for field in components]
                residual = net - gross - funding + fee + liquidation_fee
                if abs(residual) > Decimal('0.00000001'):
                    report['pnl_residuals'].append(dict(record_id=row['record_id'], market_type=row['market_type'], residual=float(residual)))
    for state in store.rows('sync_state'):
        if state['trader_id'] == trader_id and state['status'] == 'error':
            report['sync_errors'].append(dict(resource=state['resource'], market_type=state['market_type'], error=state['error']))
    report['coverage_status'] = 'observed' if successful_responses or any(report['counts'].values()) else 'uncollected'
    report['incomplete'] = bool(report['coverage_status'] == 'uncollected' or report['sync_errors'] or report['hidden_history'] or report['cursor_errors'] or report['timestamp_errors'] or any(count for fields in report['missing_fields'].values() for count in fields.values()))
    return report
