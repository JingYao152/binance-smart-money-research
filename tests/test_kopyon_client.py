import pytest
from smartmoney_research.clients.http import HttpResponse, APIError, PaginationError, HiddenDataError
from smartmoney_research.clients.kopyon import KopyonClient


class Transport:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def client(transport, **kwargs):
    return KopyonClient(transport=transport, min_interval=0, **kwargs)


def test_cursor_history_preserves_query_and_omits_missing_key(monkeypatch):
    monkeypatch.delenv('KOPYON_KEY', raising=False)
    transport = Transport(HttpResponse(200, {'data': {'positions': [{'id': 'fake-a'}], 'nextCursor': 'fake-next'}}), HttpResponse(200, {'data': {'positions': [{'id': 'fake-b'}]}}))
    assert list(client(transport).iter_history('positionHistory', 'fake-trader', 'UM', 100, 200, 20)) == [{'id': 'fake-a'}, {'id': 'fake-b'}]
    assert transport.calls[1][1]['params'] == {'traderId': 'fake-trader', 'marketType': 'UM', 'startTime': 100, 'endTime': 200, 'rows': 20, 'cursor': 'fake-next'}
    assert 'X-API-KEY' not in transport.calls[0][1]['headers']
    assert transport.calls[0][0].endswith('/v5/binance-smart-money/positionHistory')


def test_429_retry_after_and_auth():
    transport = Transport(HttpResponse(429, {'error': {'code': 'RATE', 'message': 'wait'}}, {'Retry-After': '3'}), HttpResponse(200, {'data': {'name': 'synthetic'}}))
    delays = []
    assert client(transport, api_key='fake-key', sleep=delays.append).get_profile('fake')['name'] == 'synthetic'
    assert delays == [3]
    assert transport.calls[0][1]['headers']['X-API-KEY'] == 'fake-key'


def test_5xx_bounded_backoff():
    transport = Transport(*(HttpResponse(503, {'error': {'code': 'DOWN', 'message': 'unavailable'}}) for _ in range(3)))
    delays = []
    with pytest.raises(APIError) as err:
        client(transport, max_retries=2, sleep=delays.append).get_profile('fake')
    assert err.value.status_code == 503
    assert err.value.code == 'DOWN'
    assert delays == [1, 2]
    assert len(transport.calls) == 3


def test_nonretryable_error_preserves_status():
    transport = Transport(HttpResponse(403, {'error': {'code': 'DENIED', 'message': 'scope'}}))
    with pytest.raises(APIError) as err:
        client(transport).get_profile('fake')
    assert err.value.status_code == 403
    assert err.value.code == 'DENIED'
    assert len(transport.calls) == 1


def test_repeat_cursor_raises_before_duplicate_page_rows():
    transport = Transport(*(HttpResponse(200, {'data': {'orders': [{'id': 'fake'}], 'nextCursor': 'same'}}) for _ in range(2)))
    history = client(transport).iter_history('orderHistory', 'fake', 'CM', None, None, 20)
    assert next(history) == {'id': 'fake'}
    with pytest.raises(PaginationError):
        next(history)


def test_public_resource_shapes_and_parameters():
    transport = Transport(HttpResponse(200, {'data': {'traders': [{'name': 'synthetic'}], 'hasMore': False}}), HttpResponse(200, {'data': {'points': [{'timestamp': 1, 'value': 2}]}}), HttpResponse(200, {'data': {'positions': [], 'total': 0, 'hidden': True}}))
    api = client(transport)
    assert api.search_trader('synthetic') == [{'name': 'synthetic'}]
    assert api.get_chart('fake', '1Y', 'PNL') == [{'timestamp': 1, 'value': 2}]
    assert transport.calls[1][1]['params']['chartDataType'] == 'PNL'
    assert api.get_open_positions('fake', 'UM')['hidden'] is True


def test_hidden_history_raises():
    transport = Transport(HttpResponse(200, {'data': {'positions': [], 'hidden': True}}))
    with pytest.raises(HiddenDataError):
        list(client(transport).iter_history('positionHistory', 'fake', 'UM', None, None, 20))


def test_history_rejects_arbitrary_routes_and_excess_rows():
    for resource, rows in [('private', 20), ('orderHistory', 21)]:
        with pytest.raises(ValueError):
            list(client(Transport()).iter_history(resource, 'fake', 'UM', None, None, rows))


def test_throttle_and_response_hook():
    transport = Transport(*(HttpResponse(200, {'data': {}}) for _ in range(2)))
    delays, captured = [], []
    api = KopyonClient(transport=transport, min_interval=2, clock=lambda: 0, sleep=delays.append, response_hook=lambda *args: captured.append(args))
    api.get_profile('fake')
    api.get_profile('fake')
    assert delays == [2]
    assert len(captured) == 2
    assert captured[0][1] == {'traderId': 'fake'}


def test_search_follows_has_more():
    transport = Transport(HttpResponse(200, {'data': {'traders': [{'name': 'synthetic-a'}], 'hasMore': True}}), HttpResponse(200, {'data': {'traders': [{'name': 'synthetic-b'}], 'hasMore': False}}))
    assert len(client(transport).search_trader('synthetic')) == 2
    assert transport.calls[1][1]['params']['page'] == 2


def test_open_positions_collects_total_and_preserves_hidden():
    transport = Transport(HttpResponse(200, {'data': {'positions': [{'id': 'fake-a'}], 'total': 2, 'hidden': False}}), HttpResponse(200, {'data': {'positions': [{'id': 'fake-b'}], 'total': 2, 'hidden': False}}))
    data = client(transport).get_open_positions('fake', 'UM')
    assert data == {'positions': [{'id': 'fake-a'}, {'id': 'fake-b'}], 'total': 2, 'hidden': False}
    assert transport.calls[1][1]['params']['page'] == 2


def test_success_status_error_envelope_keeps_status():
    transport = Transport(HttpResponse(200, {'error': {'code': 'SYNTHETIC', 'message': 'application failure'}}))
    with pytest.raises(APIError) as err:
        client(transport).get_profile('fake')
    assert err.value.status_code == 200
    assert err.value.code == 'SYNTHETIC'


def test_later_hidden_position_page_preserves_flag():
    transport = Transport(HttpResponse(200, {'data': {'positions': [{'id': 'fake-a'}], 'total': 2, 'hidden': False}}), HttpResponse(200, {'data': {'positions': [], 'total': 2, 'hidden': True}}))
    data = client(transport).get_open_positions('fake', 'UM')
    assert data['hidden'] is True
    assert data['positions'] == [{'id': 'fake-a'}]


def test_open_position_empty_page_cannot_satisfy_total():
    transport = Transport(HttpResponse(200, {'data': {'positions': [], 'total': 2, 'hidden': False}}))
    with pytest.raises(PaginationError):
        client(transport).get_open_positions('fake', 'UM')


def test_auditable_page_iterators_retain_full_response_envelopes():
    profile = {'data': {'name': 'Synthetic'}, 'meta': {'revision': 'profile-v1'}}
    open_positions = {'data': {'positions': [], 'total': 0, 'hidden': False}, 'meta': {'revision': 'open-v1'}}
    history = {'data': {'orders': [{'symbol': 'SYNUSDT'}]}, 'meta': {'revision': 'history-v1'}}
    api = client(Transport(HttpResponse(200, profile), HttpResponse(200, open_positions), HttpResponse(200, history)))

    assert api.get_profile_response('fake') == profile
    assert list(api.iter_open_positions_pages('fake', 'UM')) == [({'traderId': 'fake', 'marketType': 'UM', 'page': 1, 'rows': 20}, open_positions)]
    assert list(api.iter_history_pages('orderHistory', 'fake', 'UM', None, None)) == [({'traderId': 'fake', 'marketType': 'UM', 'rows': 20}, history)]
