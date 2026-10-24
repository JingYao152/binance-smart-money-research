import pytest
from smartmoney_research.config import load_config


def test_defaults_and_no_secrets(monkeypatch):
    monkeypatch.delenv('KOPYON_KEY', raising=False)
    cfg = load_config(environ={})
    assert cfg.kopyon_key is None
    assert cfg.collection.days == 365
    assert cfg.http.timeout == 30
    assert cfg.http.min_interval >= 1
    assert cfg.collection.market_type == 'UM'
    assert cfg.collection.symbols == ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
    assert cfg.collection.intervals == ('1h',)


def test_toml_and_environment(tmp_path):
    path = tmp_path / 'settings.toml'
    path.write_text('[http]\ntimeout=12\nmin_interval=2.0\n[collection]\ndays=90\nmarket_type="CM"\nsymbols=["BTCUSDT", "ETHUSDT"]\nintervals=["15m", "1h"]\n')
    cfg = load_config(path, environ={'KOPYON_KEY': 'synthetic-secret'})
    assert cfg.http.timeout == 12
    assert cfg.collection.days == 90
    assert cfg.collection.market_type == 'CM'
    assert cfg.collection.symbols == ('BTCUSDT', 'ETHUSDT')
    assert cfg.collection.intervals == ('15m', '1h')
    assert cfg.kopyon_key == 'synthetic-secret'
    assert 'synthetic-secret' not in repr(cfg)


@pytest.mark.parametrize('content', ['[http]\ntimeout=0', '[http]\nmax_retries=-1', '[http]\nmin_interval=-1', '[collection]\ndays=0', '[collection]\nmarket_type="SPOT"', '[collection]\nsymbols="BTCUSDT"', '[collection]\nintervals=["2M"]'])
def test_invalid_config(tmp_path, content):
    path = tmp_path / 'bad.toml'
    path.write_text(content)
    with pytest.raises(ValueError):
        load_config(path, environ={})
