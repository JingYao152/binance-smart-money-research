"""Binance USDⓈ-M documented public market endpoints only."""
from .http import HttpClient, PaginationError

BASE_URL = 'https://fapi.binance.com'


class BinanceClient:
    def __init__(self, **http_options):
        self.http = HttpClient(**http_options)

    def _page_batches(self, resource: str, symbol: str, start_ms: int, end_ms: int, limit: int, timestamp, **extra):
        if start_ms > end_ms:
            raise ValueError('start_ms must be <= end_ms')
        cursor = start_ms
        while cursor <= end_ms:
            params = dict(symbol=symbol, startTime=cursor, endTime=end_ms, limit=limit, **extra)
            batch = self.http.get(f'{BASE_URL}/fapi/v1/{resource}', params)
            yield params, batch
            if not isinstance(batch, list):
                raise ValueError('Expected a market data list')
            if not batch:
                return
            times = [int(timestamp(row)) for row in batch]
            if times != sorted(times) or times[0] < cursor:
                raise PaginationError(f'Non-advancing or unordered {resource} page')
            cursor = times[-1] + 1

    def _pages(self, resource: str, symbol: str, start_ms: int, end_ms: int, limit: int, timestamp, **extra):
        for _, batch in self._page_batches(resource, symbol, start_ms, end_ms, limit, timestamp, **extra):
            if not isinstance(batch, list):
                raise ValueError('Expected a market data list')
            if not batch:
                return
            times = [int(timestamp(row)) for row in batch]
            for row, stamp in zip(batch, times):
                if stamp <= end_ms:
                    yield row

    def iter_klines_pages(self, symbol: str, interval: str, start_ms: int, end_ms: int):
        """Yield exact Binance K-line request params and raw response pages."""
        yield from self._page_batches('klines', symbol, start_ms, end_ms, 1500, lambda x: x[0], interval=interval)

    def iter_funding_pages(self, symbol: str, start_ms: int, end_ms: int):
        """Yield exact Binance funding request params and raw response pages."""
        yield from self._page_batches('fundingRate', symbol, start_ms, end_ms, 1000, lambda x: x['fundingTime'])

    def iter_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int):
        for row in self._pages('klines', symbol, start_ms, end_ms, 1500, lambda x: x[0], interval=interval):
            if len(row) != 12:
                raise ValueError('Expected a 12-field Binance kline')
            yield dict(symbol=symbol, interval=interval, open_time=int(row[0]), open=float(row[1]), high=float(row[2]), low=float(row[3]), close=float(row[4]), volume=float(row[5]), close_time=int(row[6]), quote_volume=float(row[7]), trades=int(row[8]), taker_buy_base_volume=float(row[9]), taker_buy_quote_volume=float(row[10]))

    def iter_funding_rates(self, symbol: str, start_ms: int, end_ms: int):
        for row in self._pages('fundingRate', symbol, start_ms, end_ms, 1000, lambda x: x['fundingTime']):
            result = dict(symbol=row['symbol'], funding_time=int(row['fundingTime']), funding_rate=float(row['fundingRate']))
            if 'markPrice' in row:
                result['mark_price'] = float(row['markPrice'])
            if 'rateType' in row:
                result['rate_type'] = row['rateType']
            yield result
