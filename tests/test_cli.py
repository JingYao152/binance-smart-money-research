import json

from smartmoney_research.cli import build_parser, main, parse_trader_id


def test_cli_lists_all_requested_commands():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {"sync", "inspect", "analyze", "backtest", "skill", "export", "dashboard"} <= set(choices)


def test_dry_run_reports_365_day_window_without_network(capsys):
    status = main(["sync", "--trader-id", "synthetic-id", "--dry-run", "--now-ms", "1000000000000"])
    out = json.loads(capsys.readouterr().out)
    assert status == 0
    assert out["start_ms"] == 1000000000000 - 365 * 24 * 60 * 60 * 1000
    assert out["trader_id"] == "synthetic-id"
    assert out["market_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert out["market_intervals"] == ["1h"]


def test_dry_run_uses_configured_collection_days(tmp_path, capsys):
    config_path = tmp_path / "custom.toml"
    config_path.write_text('[collection]\ndays = 42\n[storage]\npath = "custom.duckdb"\n', encoding="utf-8")
    status = main(["sync", "--config", str(config_path), "--trader-id", "synthetic-id", "--dry-run", "--now-ms", "1000000000000"])
    out = json.loads(capsys.readouterr().out)
    assert status == 0
    assert out["start_ms"] == 1000000000000 - 42 * 24 * 60 * 60 * 1000


def test_trader_id_parser_accepts_kopyon_id_and_rejects_square_creator_slug():
    assert parse_trader_id("123456789") == "123456789"
    assert parse_trader_id("https://example.invalid/trader/123456789") == "123456789"
    assert parse_trader_id("https://www.binance.com/zh-CN/square/profile/square-creator-example") is None


def test_dashboard_dispatch_does_not_sync(monkeypatch):
    called = []
    monkeypatch.setattr("smartmoney_research.cli._run_dashboard", lambda args: called.append(args))
    assert main(["dashboard", "--database", "local.duckdb"]) == 0
    assert len(called) == 1


def test_dashboard_uses_database_path_from_config(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.toml"
    config_path.write_text('[storage]\npath = "custom.duckdb"\n', encoding="utf-8")
    called = []
    monkeypatch.setattr("smartmoney_research.cli._run_dashboard", lambda args: called.append(args.database))
    assert main(["dashboard", "--config", str(config_path)]) == 0
    assert called == ["custom.duckdb"]


def test_inspect_and_export_use_configured_database(tmp_path, capsys):
    config_path = tmp_path / "custom.toml"
    database = tmp_path / "configured.duckdb"
    config_path.write_text(f'[storage]\npath = "{database}"\n', encoding="utf-8")
    assert main(["inspect", "--config", str(config_path), "--trader-id", "synthetic-id"]) == 0
    capsys.readouterr()
    output_path = tmp_path / "schema.json"
    assert main(["export", "--config", str(config_path), "--table", "schema_version", "--format", "json", "--path", str(output_path)]) == 0
    assert database.exists()
    assert output_path.exists()



def _local_research_db(tmp_path):
    import math
    from datetime import datetime, timezone
    from smartmoney_research.storage import ResearchStore
    path = tmp_path / 'research.duckdb'
    with ResearchStore(path) as store:
        store.initialize()
        for symbol, base in [('ETHUSDT', 100), ('BTCUSDT', 1000)]:
            bars = []
            for index in range(80):
                price = base * (1 + .1 * math.sin(index / 4))
                bars.append(dict(symbol=symbol, interval='1h', open_time=index * 3600000,
                                 close_time=(index + 1) * 3600000 - 1, open=price,
                                 close=price, high=price * 1.01, low=price * .99, volume=100,
                                 fetched_at=datetime.fromtimestamp(80 * 3600, timezone.utc)))
            store.upsert_history('klines', bars)
            store.upsert_history('funding_rates', [dict(symbol=symbol,
                                 funding_time=40 * 3600000, funding_rate=.001,
                                 mark_price=base, rate_type='realized')])
        store.upsert_history('position_history', [dict(trader_id='synthetic', market_type='UM',
                             record_id='fixture', symbol='ETHUSDT', side='LONG',
                             open_time=40 * 3600000 - 1, close_time=55 * 3600000)])
    return path


def test_local_analyze_has_strict_as_of_indicators_without_writes(tmp_path, capsys, monkeypatch):
    import hashlib
    import socket
    path = _local_research_db(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    assert main(['analyze', '--database', str(path), '--trader-id', 'synthetic']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['evidence']['trader_id'] == 'synthetic'
    events = output['trade_events']
    assert len(events) == 2
    assert all(row['bar_close_time'] < row['event_time'] for row in events)
    assert all(row['rsi'] is not None for row in events)
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_backtest_costs_funding_benchmark_and_train_only_selection(tmp_path, capsys, monkeypatch):
    import hashlib
    import socket
    path = _local_research_db(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    assert main(['backtest', '--database', str(path), '--trader-id', 'synthetic',
                 '--symbol', 'ETHUSDT', '--rule', 'ema_crossover', '--fast', '3', '--slow', '7',
                 '--parameter-grid', '{"fast":[3,4],"slow":[7]}', '--train-rows', '40', '--test-rows', '20']) == 0
    output = json.loads(capsys.readouterr().out)
    run = output['symbols']['ETHUSDT']
    assert run['single']['assumptions']['leverage'] == 1
    assert run['single']['assumptions']['fee_rate'] > 0
    assert run['single']['benchmark']['available'] is True
    assert run['single']['funding']['rows_skipped_other_symbol'] == 1
    assert run['sensitivity']['combinations_evaluated'] == 2
    assert run['walk_forward']
    for fold in run['walk_forward']:
        assert fold['train_end_time'] < fold['test_start_time']
        assert all(item['evaluated_through'] <= fold['train_end_time']
                   for item in fold['training_grid']['results'])
    assert before == hashlib.sha256(path.read_bytes()).hexdigest()


def test_local_workflows_require_existing_database(tmp_path, capsys):
    path = tmp_path / 'absent.duckdb'
    for command in ['analyze', 'backtest']:
        assert main([command, '--database', str(path), '--trader-id', 'synthetic']) == 2
        capsys.readouterr()
    assert not path.exists()


def test_unverified_future_bars_do_not_enter_strategy_or_btc_backtests(tmp_path, capsys):
    from datetime import datetime, timedelta, timezone
    from smartmoney_research.storage import ResearchStore
    path = _local_research_db(tmp_path)
    future_ms = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp() * 1000)
    with ResearchStore(path) as store:
        for symbol in ['ETHUSDT', 'BTCUSDT']:
            store.upsert_history('klines', [dict(symbol=symbol, interval='1h', open_time=future_ms - 1000,
                                 close_time=future_ms, fetched_at=None, open=999999,
                                 close=999999, high=999999, low=999999, volume=100)])
    assert main(['backtest', '--database', str(path), '--trader-id', 'synthetic',
                 '--symbol', 'ETHUSDT', '--fast', '3', '--slow', '7',
                 '--train-rows', '40', '--test-rows', '20']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['symbols']['ETHUSDT']['single']['equity'][-1]['time'] == 79 * 3600000
    assert result['symbols']['ETHUSDT']['single']['benchmark']['end_time'] == 79 * 3600000
    for symbol in ['ETHUSDT', 'BTCUSDT']:
        coverage = result['market_bar_coverage'][symbol]
        assert coverage['unavailable_rows'] == 1
        assert coverage['missing_retrieval'] == 1
        assert coverage['future_close'] == 1
        assert coverage['verified_rows'] == 80
