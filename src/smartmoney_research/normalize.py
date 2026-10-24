"""Lossless source copies with explicitly mapped normalized fields.

Fallback identities hash canonical JSON of trader, market, resource and the full
source record. Identical content cannot distinguish distinct identical orders;
changed source content produces another identity. No provider order ID is assumed.
Provenance is excluded so repeated fetches deduplicate. Raw values remain intact.
"""
from copy import deepcopy
from hashlib import sha256
import json


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _base(trader_id, market_type, record, resource, raw_response_id, fetched_at):
    provider_id = record.get('positionId') if resource == 'position' else None
    identity = str(provider_id) if provider_id is not None else sha256(canonical_json([trader_id, market_type, resource, record]).encode()).hexdigest()
    return dict(trader_id=trader_id, market_type=market_type, record_id=identity,
                id_kind='provider' if provider_id is not None else 'content_sha256',
                source='kopyon', raw_response_id=raw_response_id, fetched_at=fetched_at,
                source_details=deepcopy(record.get('sourceDetails')),
                raw_fields=deepcopy(record))


def normalize_position(trader_id: str, market_type: str, record: dict, *, raw_response_id=None, fetched_at=None):
    row = _base(trader_id, market_type, record, 'position', raw_response_id, fetched_at)
    mapping = dict(position_id='positionId', symbol='symbol', side='side', status='status', size='size',
                   entry_price='entryPrice', mark_price='markPrice', close_price='closePrice', leverage='leverage',
                   margin='margin', unrealized_pnl='unrealizedPnl', open_time='openedAt', close_time='closedAt', updated_at='updatedAt', margin_mode='marginMode',
                   opened_volume='openedVolume', closed_volume='closedVolume', gross_pnl='grossPnl',
                   funding_pnl='fundingPnl', trading_fee='tradingFee', liquidation_fee='liquidationFee', net_pnl='netPnl')
    row.update({key: record.get(source) for key, source in mapping.items()})
    return row


def normalize_order(trader_id: str, market_type: str, record: dict, *, raw_response_id=None, fetched_at=None):
    row = _base(trader_id, market_type, record, 'order', raw_response_id, fetched_at)
    mapping = dict(symbol='symbol', side='side', position_side='positionSide', price='price', quantity='quantity',
                   quote_quantity='quoteQuantity', timestamp='createdAt', created_at='createdAt', updated_at='updatedAt', commission='commission', realized_pnl='realizedPnl')
    row.update({key: record.get(source) for key, source in mapping.items()})
    return row
