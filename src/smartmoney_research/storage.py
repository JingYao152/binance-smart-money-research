"""Local DuckDB evidence and current normalized state. All timestamps use UTC.

Snapshot methods append evidence; upserts replace supplied fields for a scoped key.
Use normalized rows for positions/orders and attach the originating raw response ID
and fetched_at. Missing source values remain NULL, including provenance unavailable
at normalization time. Checkpoints are explicit: callers advance a watermark only
when a sync completes. Arrays, objects and scalar JSON responses are supported.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from uuid import uuid4

import duckdb

from .normalize import canonical_json

_PROVENANCE = {'source': 'VARCHAR', 'raw_response_id': 'VARCHAR', 'fetched_at': 'TIMESTAMPTZ'}
_IDENTITY = {'trader_id': 'VARCHAR', 'market_type': 'VARCHAR', 'record_id': 'VARCHAR', 'id_kind': 'VARCHAR', **_PROVENANCE, 'source_details': 'JSON', 'raw_fields': 'JSON'}
_POSITION = {**_IDENTITY, **{key: 'VARCHAR' for key in ('position_id', 'symbol', 'side', 'status', 'margin_mode')}, **{key: 'DOUBLE' for key in ('size', 'entry_price', 'mark_price', 'close_price', 'leverage', 'margin', 'unrealized_pnl', 'opened_volume', 'closed_volume', 'gross_pnl', 'funding_pnl', 'trading_fee', 'liquidation_fee', 'net_pnl')}, 'open_time': 'BIGINT', 'close_time': 'BIGINT', 'updated_at': 'BIGINT'}
_SCHEMAS = {
    'traders': {'trader_id': 'VARCHAR', 'name': 'VARCHAR', **_PROVENANCE, 'raw_fields': 'JSON'},
    'profiles': {'snapshot_id': 'VARCHAR', 'trader_id': 'VARCHAR', **_PROVENANCE, 'raw_fields': 'JSON'},
    'charts': {'snapshot_id': 'VARCHAR', 'trader_id': 'VARCHAR', 'period': 'VARCHAR', 'chart_type': 'VARCHAR', 'timestamp': 'BIGINT', 'value': 'DOUBLE', **_PROVENANCE, 'raw_fields': 'JSON'},
    'open_positions': _POSITION,
    'position_history': _POSITION,
    'order_history': {**_IDENTITY, **{k: 'VARCHAR' for k in ('symbol', 'side', 'position_side')}, **{k: 'DOUBLE' for k in ('price', 'quantity', 'quote_quantity', 'commission', 'realized_pnl')}, 'timestamp': 'BIGINT', 'created_at': 'BIGINT', 'updated_at': 'BIGINT'},
    'klines': {'symbol': 'VARCHAR', 'interval': 'VARCHAR', 'open_time': 'BIGINT', 'close_time': 'BIGINT', **{k: 'DOUBLE' for k in ('open', 'high', 'low', 'close', 'volume', 'quote_volume', 'taker_buy_base_volume', 'taker_buy_quote_volume')}, 'trades': 'BIGINT', **_PROVENANCE, 'raw_fields': 'JSON'},
    'funding_rates': {'symbol': 'VARCHAR', 'funding_time': 'BIGINT', 'funding_rate': 'DOUBLE', 'mark_price': 'DOUBLE', 'rate_type': 'VARCHAR', **_PROVENANCE, 'raw_fields': 'JSON'},
    'sync_state': {'trader_id': 'VARCHAR', 'market_type': 'VARCHAR', 'resource': 'VARCHAR', 'watermark': 'BIGINT', 'cursor': 'VARCHAR', 'status': 'VARCHAR', 'error': 'VARCHAR', 'updated_at': 'TIMESTAMPTZ'},
}
_KEYS = {'traders': ('trader_id',), 'profiles': ('snapshot_id',), 'charts': ('snapshot_id',),
         **{t: ('trader_id', 'market_type', 'record_id') for t in ('open_positions', 'position_history', 'order_history')},
         'klines': ('symbol', 'interval', 'open_time'), 'funding_rates': ('symbol', 'funding_time'),
         'sync_state': ('trader_id', 'market_type', 'resource')}


class ResearchStore:
    def __init__(self, path, *, read_only=False):
        self.connection = duckdb.connect(str(path), read_only=read_only)

    def initialize(self):
        self.connection.execute('CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)')
        versions = self.connection.execute('SELECT version FROM schema_version').fetchall()
        if versions and versions != [(1,)]:
            raise ValueError('Unsupported database schema version')
        self.connection.execute('INSERT INTO schema_version VALUES (1) ON CONFLICT DO NOTHING')
        self.connection.execute('''CREATE TABLE IF NOT EXISTS raw_responses (
            raw_response_id VARCHAR PRIMARY KEY, source VARCHAR, endpoint VARCHAR,
            params JSON, payload JSON, status INTEGER, fetched_at TIMESTAMPTZ, content_hash VARCHAR)''')
        for table, fields in _SCHEMAS.items():
            columns = ', '.join(f'"{key}" {kind}' for key, kind in fields.items())
            keys = ', '.join(f'"{key}"' for key in _KEYS[table])
            self.connection.execute(f'CREATE TABLE IF NOT EXISTS {table} ({columns}, PRIMARY KEY ({keys}))')
        return self

    def save_raw(self, source: str, endpoint: str, params: dict, payload: object, status: int, fetched_at: datetime):
        """Append one retrieval. content_hash is SHA-256 of canonical payload JSON."""
        encoded = canonical_json(payload)
        identity = str(uuid4())
        self.connection.execute('INSERT INTO raw_responses VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                                [identity, source, endpoint, canonical_json(params), encoded, status, fetched_at, sha256(encoded.encode()).hexdigest()])
        return identity

    def rows(self, table: str):
        self._validate_table(table)
        cursor = self.connection.execute(f'SELECT * FROM {table}')
        names = [entry[0] for entry in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def table_schema(self, table: str):
        """Return column names and DuckDB types without reading table rows."""
        self._validate_table(table)
        return [(row[0], row[1]) for row in self.connection.execute(f'DESCRIBE SELECT * FROM {table}').fetchall()]

    @staticmethod
    def _validate_table(table: str):
        if table not in {*_SCHEMAS, 'raw_responses', 'schema_version'}:
            raise ValueError('Unsupported table')

    def upsert_history(self, table: str, records: list[dict]):
        with self._transaction():
            self._upsert(table, records)

    def _upsert(self, table, records):
        if table not in _SCHEMAS:
            raise ValueError('Unsupported upsert table')
        prepared = []
        for record in records:
            if set(record) - _SCHEMAS[table].keys():
                raise ValueError('Unknown columns')
            if any(record.get(key) is None for key in _KEYS[table]):
                raise ValueError('Missing record key')
            columns = list(record)
            values = [canonical_json(record[k]) if _SCHEMAS[table][k] == 'JSON' and record[k] is not None else record[k] for k in columns]
            quoted = ', '.join(f'"{k}"' for k in columns)
            marks = ', '.join('?' for _ in columns)
            changes = ', '.join(f'"{k}" = excluded."{k}"' for k in columns if k not in _KEYS[table])
            conflict = 'DO UPDATE SET ' + changes if changes else 'DO NOTHING'
            prepared.append((f'INSERT INTO {table} ({quoted}) VALUES ({marks}) ON CONFLICT {conflict}', values))
        for sql, values in prepared:
            self.connection.execute(sql, values)

    @contextmanager
    def _transaction(self):
        self.connection.execute('BEGIN TRANSACTION')
        try:
            yield
            self.connection.execute('COMMIT')
        except Exception:
            self.connection.execute('ROLLBACK')
            raise

    def save_profile(self, trader_id, profile, *, raw_response_id=None, fetched_at=None, source='kopyon'):
        identity = str(uuid4())
        self.upsert_history('profiles', [dict(snapshot_id=identity, trader_id=trader_id, raw_fields=profile, source=source, raw_response_id=raw_response_id, fetched_at=fetched_at)])
        return identity

    def save_chart_points(self, trader_id, period, chart_type, points, *, raw_response_id=None, fetched_at=None, source='kopyon'):
        self.upsert_history('charts', [dict(snapshot_id=str(uuid4()), trader_id=trader_id, period=period, chart_type=chart_type, timestamp=p.get('timestamp'), value=p.get('value'), raw_fields=p, source=source, raw_response_id=raw_response_id, fetched_at=fetched_at) for p in points])

    def replace_open_positions(self, trader_id, market_type, records):
        if any(r.get('trader_id') != trader_id or r.get('market_type') != market_type for r in records):
            raise ValueError('Open-position scope mismatch')
        with self._transaction():
            self._upsert('open_positions', records)
            ids = [r['record_id'] for r in records]
            exclusion = ' AND record_id NOT IN (' + ','.join('?' for _ in ids) + ')' if ids else ''
            self.connection.execute('DELETE FROM open_positions WHERE trader_id = ? AND market_type = ?' + exclusion, [trader_id, market_type, *ids])

    def save_checkpoint(self, trader_id, market_type, resource, *, watermark=None, cursor=None, status, error=None, updated_at=None):
        self.upsert_history('sync_state', [dict(trader_id=trader_id, market_type=market_type, resource=resource, watermark=watermark, cursor=cursor, status=status, error=error, updated_at=updated_at or datetime.now(timezone.utc))])

    def get_checkpoint(self, trader_id, market_type, resource):
        return next((row for row in self.rows('sync_state') if (row['trader_id'], row['market_type'], row['resource']) == (trader_id, market_type, resource)), None)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
