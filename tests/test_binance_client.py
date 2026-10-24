import pytest
from smartmoney_research.clients.http import HttpResponse, PaginationError
from smartmoney_research.clients.binance import BinanceClient


class Transport:
    def __init__(self, *payloads):
        self.payloads, self.calls = iter(payloads), []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return HttpResponse(200, next(self.payloads))


def bar(time):
    return [time, '10', '12', '9', '11', '3', time+59, '32', 4, '1', '11', '0']


def test_kline_parse_and_advance():
    transport = Transport([bar(100)], [bar(200)], [])
    rows = list(BinanceClient(transport=transport, min_interval=0).iter_klines('TESTUSDT', '1m', 100, 300))
    assert len(rows) == 2
    assert rows[0] == dict(symbol='TESTUSDT', interval='1m', open_time=100, open=10.0, high=12.0, low=9.0, close=11.0, volume=3.0, close_time=159, quote_volume=32.0, trades=4, taker_buy_base_volume=1.0, taker_buy_quote_volume=11.0)
    assert transport.calls[1][1]['params']['startTime'] == 101
    assert transport.calls[0][0] == 'https://fapi.binance.com/fapi/v1/klines'
    assert transport.calls[0][1]['params']['limit'] <= 1500
    assert transport.calls[0][1]['headers'] == {}


def test_funding_parse_and_advance_preserves_missing():
    transport = Transport([{'symbol': 'TESTUSDT', 'fundingTime': 100, 'fundingRate': '0.001'}], [])
    rows = list(BinanceClient(transport=transport, min_interval=0).iter_funding_rates('TESTUSDT', 100, 300))
    assert rows == [{'symbol': 'TESTUSDT', 'funding_time': 100, 'funding_rate': 0.001}]
    assert transport.calls[1][1]['params']['startTime'] == 101
    assert transport.calls[0][1]['params']['limit'] <= 1000


def test_stale_market_page_raises():
    transport = Transport([bar(100)], [bar(100)])
    with pytest.raises(PaginationError):
        list(BinanceClient(transport=transport, min_interval=0).iter_klines('TESTUSDT', '1m', 100, 300))


def test_invalid_time_range():
    with pytest.raises(ValueError):
        list(BinanceClient(transport=Transport(), min_interval=0).iter_klines('TESTUSDT', '1m', 300, 100))


def test_raw_page_iterators_preserve_exact_payload_and_request_params():
    raw_bars = [bar(100), bar(200)]
    raw_funding = [{'symbol': 'TESTUSDT', 'fundingTime': 100, 'fundingRate': '0.001', 'markPrice': '11'}]
    transport = Transport(raw_bars, [], raw_funding, [])
    api = BinanceClient(transport=transport, min_interval=0)

    kline_pages = list(api.iter_klines_pages('TESTUSDT', '1m', 100, 300))
    funding_pages = list(api.iter_funding_pages('TESTUSDT', 100, 300))

    assert kline_pages[0] == ({'symbol': 'TESTUSDT', 'interval': '1m', 'startTime': 100, 'endTime': 300, 'limit': 1500}, raw_bars)
    assert funding_pages[0] == ({'symbol': 'TESTUSDT', 'startTime': 100, 'endTime': 300, 'limit': 1000}, raw_funding)
