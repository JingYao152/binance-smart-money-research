"""Kopyon v5 routes documented at https://kopyon.com/binance-smart-money-api/."""
import os
from .http import HttpClient, PaginationError, HiddenDataError

BASE_URL = 'https://api.kopyon.com/v5/binance-smart-money'


class KopyonClient:
    def __init__(self, *, api_key: str | None = None, **http_options):
        self.http = HttpClient(**http_options)
        key = api_key if api_key is not None else os.environ.get('KOPYON_KEY')
        self._headers = {'X-API-KEY': key} if key else {}

    def _get_envelope(self, resource: str, params: dict) -> dict:
        response = self.http.get(f'{BASE_URL}/{resource}', params, self._headers)
        if not isinstance(response, dict) or 'data' not in response:
            raise ValueError(f'Expected a Kopyon data envelope for {resource}')
        return response

    def _get(self, resource: str, params: dict) -> dict:
        return self._get_envelope(resource, params)['data']

    def search_trader(self, name: str) -> list[dict]:
        matches, page = [], 1
        while True:
            data = self._get('traderSearch', {'name': name, 'page': page, 'rows': 20})
            rows = data['traders']
            matches.extend(rows)
            if not data.get('hasMore'):
                return matches
            if not rows:
                raise PaginationError('traderSearch hasMore with empty page')
            page += 1

    def get_profile(self, trader_id: str) -> dict:
        return self.get_profile_response(trader_id)['data']

    def get_profile_response(self, trader_id: str) -> dict:
        return self._get_envelope('traderProfile', {'traderId': trader_id})

    def get_chart(self, trader_id: str, period: str, chart_type: str) -> list[dict]:
        return self.get_chart_response(trader_id, period, chart_type)['data']['points']

    def get_chart_response(self, trader_id: str, period: str, chart_type: str) -> dict:
        return self._get_envelope('traderChart', {'traderId': trader_id, 'period': period, 'chartDataType': chart_type})

    def get_open_positions(self, trader_id: str, market_type: str) -> dict:
        result, positions, page = {}, [], 1
        seen = set()
        for _, envelope in self.iter_open_positions_pages(trader_id, market_type):
            data = envelope['data']
            result.update(data)
            batch = data['positions']
            signature = repr(batch)
            if batch and signature in seen:
                raise PaginationError('Repeated openPositions page')
            seen.add(signature)
            positions.extend(batch)
            result['positions'] = positions
            if data.get('hidden') or len(positions) >= data.get('total', len(positions)):
                return result
            if not batch:
                raise PaginationError('Empty openPositions page before total')

    def iter_open_positions_pages(self, trader_id: str, market_type: str):
        if market_type not in ('UM', 'CM'):
            raise ValueError('market_type must be UM or CM')
        page = 1
        seen = set()
        collected = 0
        expected_total = None
        while True:
            params = {'traderId': trader_id, 'marketType': market_type, 'page': page, 'rows': 20}
            envelope = self._get_envelope('openPositions', params)
            data = envelope['data']
            batch = data.get('positions')
            if not isinstance(batch, list):
                raise ValueError('Expected openPositions positions list')
            signature = repr(batch)
            if batch and signature in seen:
                raise PaginationError('Repeated openPositions page')
            seen.add(signature)
            yield params, envelope
            if data.get('hidden'):
                return
            expected_total = data.get('total', expected_total)
            collected += len(batch)
            if expected_total is not None and collected >= expected_total:
                return
            if not batch:
                raise PaginationError('Empty openPositions page before total')
            page += 1

    def iter_history_pages(self, resource: str, trader_id: str, market_type: str, start_ms: int | None, end_ms: int | None, rows: int = 20):
        if resource not in ('positionHistory', 'orderHistory'):
            raise ValueError('Unsupported history resource')
        if type(rows) is not int or not 1 <= rows <= 20:
            raise ValueError('rows must be between 1 and 20')
        if market_type not in ('UM', 'CM'):
            raise ValueError('market_type must be UM or CM')
        if start_ms is not None and end_ms is not None and start_ms > end_ms:
            raise ValueError('start_ms must be <= end_ms')
        params = {'traderId': trader_id, 'marketType': market_type, 'rows': rows}
        if start_ms is not None:
            params['startTime'] = start_ms
        if end_ms is not None:
            params['endTime'] = end_ms
        seen = set()
        while True:
            request_params = dict(params)
            envelope = self._get_envelope(resource, request_params)
            data = envelope['data']
            cursor = data.get('nextCursor')
            if cursor and (cursor == request_params.get('cursor') or cursor in seen):
                raise PaginationError(f'Repeated cursor in {resource}')
            yield request_params, envelope
            if not cursor:
                return
            seen.add(cursor)
            params['cursor'] = cursor

    def iter_history(self, resource: str, trader_id: str, market_type: str, start_ms: int | None, end_ms: int | None, rows: int = 20):
        for _, envelope in self.iter_history_pages(resource, trader_id, market_type, start_ms, end_ms, rows):
            data = envelope['data']
            if data.get('hidden'):
                raise HiddenDataError(resource, trader_id)
            yield from data['positions' if resource == 'positionHistory' else 'orders']
