"""Small synchronous GET transport with bounded retries and response evidence hooks."""
from dataclasses import dataclass, field
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from smartmoney_research.config import HTTPSettings


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    payload: Any
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def __call__(self, url: str, *, params: dict, headers: dict, timeout: float) -> HttpResponse: ...


class APIError(RuntimeError):
    def __init__(self, status_code: int, code: Any, message: str):
        self.status_code, self.code = status_code, code
        super().__init__(f'HTTP {status_code}, {code}: {message}')


class PaginationError(RuntimeError):
    pass


class HiddenDataError(RuntimeError):
    def __init__(self, resource: str, trader_id: str):
        self.resource, self.trader_id = resource, trader_id
        super().__init__(f'{resource} is hidden for trader {trader_id}')


def urllib_transport(url: str, *, params: dict, headers: dict, timeout: float) -> HttpResponse:
    request = Request(f'{url}?{urlencode(params)}', headers=headers, method='GET')
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            # Keep non-JSON error bodies available to the local evidence hook.
            payload = {'message': raw.decode('utf-8', errors='replace')}
        return HttpResponse(response.code, payload, dict(response.headers))


class HttpClient:
    def __init__(self, *, transport: Transport | None = None, timeout: float = 30, min_interval: float = 1, max_retries: int = 3, backoff_factor: float = 1, sleep: Callable = time.sleep, clock: Callable = time.monotonic, response_hook: Callable | None = None):
        self.settings = HTTPSettings(timeout, min_interval, max_retries, backoff_factor)
        self.transport = transport or urllib_transport
        self.sleep, self.clock = sleep, clock
        self.response_hook = response_hook
        self._last_request = None

    def get(self, url: str, params: dict, headers: dict | None = None) -> Any:
        for attempt in range(self.settings.max_retries + 1):
            if self._last_request is not None:
                delay = self.settings.min_interval - (self.clock() - self._last_request)
                if delay > 0:
                    self.sleep(delay)
            self._last_request = self.clock()
            response = self.transport(url, params=dict(params), headers=dict(headers or {}), timeout=self.settings.timeout)
            if self.response_hook:
                self.response_hook(url, dict(params), response)
            retryable = response.status_code == 429 or 500 <= response.status_code < 600
            if retryable and attempt < self.settings.max_retries:
                delay = self.settings.backoff_factor * 2 ** attempt
                retry_after = next((v for k, v in response.headers.items() if k.lower() == 'retry-after'), None)
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        try:
                            when = parsedate_to_datetime(retry_after)
                            delay = max(delay, (when - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError, OverflowError):
                            pass
                self.sleep(delay)
                continue
            payload = response.payload
            error = payload.get('error') if isinstance(payload, dict) else None
            if not 200 <= response.status_code < 300 or error:
                detail = error if isinstance(error, dict) else {}
                raise APIError(response.status_code, detail.get('code', payload.get('code') if isinstance(payload, dict) else None), str(detail.get('message', payload.get('msg', payload.get('message', 'Request failed')) if isinstance(payload, dict) else 'Request failed')))
            return payload
        raise AssertionError('unreachable')
