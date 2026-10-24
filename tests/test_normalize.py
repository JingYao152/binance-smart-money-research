from copy import deepcopy

from smartmoney_research.normalize import normalize_order, normalize_position


def test_position_keeps_identity_details_and_missing_values():
    original = {'positionId': 'position-1', 'symbol': 'SYNUSD', 'sourceDetails': {'extra': 3}, 'unknown': 'preserve', 'fundingPnl': '-2'}
    before = deepcopy(original)
    result = normalize_position('synthetic', 'UM', original)
    assert result['record_id'] == 'position-1'
    assert result['id_kind'] == 'provider'
    assert result['source_details'] == {'extra': 3}
    assert result['raw_fields'] == original == before
    assert result['trading_fee'] is None
    assert result['funding_pnl'] == '-2'


def test_order_fingerprint_is_stable_scoped_and_retains_full_record():
    original = {'symbol': 'SYNUSD', 'side': 'BUY', 'positionSide': 'LONG', 'quantity': '2', 'price': '3', 'createdAt': 1000}
    a = normalize_order('a', 'UM', original)
    b = normalize_order('a', 'UM', dict(reversed(list(original.items()))))
    assert a['record_id'] == b['record_id']
    assert a['id_kind'] == 'content_sha256'
    assert a['record_id'] != normalize_order('b', 'UM', original)['record_id']
    assert a['record_id'] != normalize_order('a', 'CM', original)['record_id']
    assert a['record_id'] != normalize_order('a', 'UM', dict(original, quantity='3'))['record_id']
    assert a['raw_fields'] == original
    assert a['timestamp'] == 1000
    assert a['commission'] is None


def test_position_without_id_has_stable_fingerprint():
    assert normalize_position('a', 'UM', {'symbol': 'SYN'})['record_id'] == normalize_position('a', 'UM', {'symbol': 'SYN'})['record_id']


def test_verified_provider_timestamp_and_margin_fields():
    pos = normalize_position('a', 'UM', {'openedAt': 1000, 'closedAt': 2000, 'updatedAt': 3000, 'marginMode': 'CROSS'})
    assert pos['open_time'] == 1000
    assert pos['close_time'] == 2000
    assert pos['updated_at'] == 3000
    assert pos['margin_mode'] == 'CROSS'
    order = normalize_order('a', 'UM', {'createdAt': 1000, 'updatedAt': 2000})
    assert order['timestamp'] == order['created_at'] == 1000
    assert order['updated_at'] == 2000
