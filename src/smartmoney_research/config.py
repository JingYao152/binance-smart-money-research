"""Typed local settings. Credentials are read only from the environment."""
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping
import os
import tomllib
import math
import re

_SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}


@dataclass(frozen=True)
class HTTPSettings:
    timeout: float = 30.0
    min_interval: float = 1.0
    max_retries: int = 3
    backoff_factor: float = 1.0

    def __post_init__(self):
        for name in ('timeout', 'min_interval', 'backoff_factor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.timeout == 0:
            raise ValueError('timeout must be positive')
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')


@dataclass(frozen=True)
class CollectionSettings:
    days: int = 365
    market_type: str = 'UM'
    overlap_ms: int = 86400000
    symbols: tuple[str, ...] = ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
    intervals: tuple[str, ...] = ('1h',)

    def __post_init__(self):
        if type(self.days) is not int or self.days <= 0:
            raise ValueError('days must be a positive integer')
        if self.market_type not in ('UM', 'CM'):
            raise ValueError('market_type must be UM or CM')
        if type(self.overlap_ms) is not int or self.overlap_ms < 0:
            raise ValueError('overlap_ms must be a nonnegative integer')
        if not isinstance(self.symbols, (tuple, list)) or not isinstance(self.intervals, (tuple, list)):
            raise ValueError('symbols and intervals must be TOML arrays')
        symbols = tuple(self.symbols)
        intervals = tuple(self.intervals)
        if not symbols or any(not isinstance(symbol, str) or not re.fullmatch(r'[A-Z0-9_]{1,30}', symbol) for symbol in symbols):
            raise ValueError('symbols must be a nonempty list of valid uppercase Binance symbols')
        if not intervals or any(not isinstance(interval, str) or interval not in _SUPPORTED_INTERVALS for interval in intervals):
            raise ValueError(f'intervals must be a nonempty list drawn from {sorted(_SUPPORTED_INTERVALS)}')
        object.__setattr__(self, 'symbols', tuple(dict.fromkeys(symbols)))
        object.__setattr__(self, 'intervals', tuple(dict.fromkeys(intervals)))


@dataclass(frozen=True)
class StorageSettings:
    path: str = 'data/research.duckdb'


@dataclass(frozen=True)
class Settings:
    http: HTTPSettings = field(default_factory=HTTPSettings)
    collection: CollectionSettings = field(default_factory=CollectionSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    kopyon_key: str | None = field(default=None, repr=False)


def load_config(path: str | Path | None = None, *, environ: Mapping[str, str] | None = None) -> Settings:
    values = {} if path is None else tomllib.loads(Path(path).read_text())
    unknown = set(values) - {'http', 'collection', 'storage'}
    if unknown:
        raise ValueError(f'Unknown settings sections: {sorted(unknown)}')
    env = os.environ if environ is None else environ
    return Settings(http=HTTPSettings(**values.get('http', {})), collection=CollectionSettings(**values.get('collection', {})), storage=StorageSettings(**values.get('storage', {})), kopyon_key=env.get('KOPYON_KEY') or None)
