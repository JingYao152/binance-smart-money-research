"""Command-line entry points for local collection and research workflows."""
import argparse
from datetime import datetime, timezone
import json
import math
import re
import sys
from urllib.parse import parse_qs, urlparse

from .config import load_config
from .exports import export_table
from .ingest import sync_trader
from .quality import check_integrity
from .storage import ResearchStore


def parse_trader_id(value: str | None) -> str | None:
    """Return a Kopyon trader ID from an ID or an explicit trader URL.

    Binance Square creator slugs identify a profile page, not the Kopyon
    trader ID, so those values are left unresolved for name search.
    """
    if not value:
        return None
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        query = parse_qs(parsed.query)
        for key in ("traderId", "trader_id", "id"):
            if query.get(key):
                return query[key][0]
        parts = [part for part in parsed.path.split("/") if part]
        if "trader" in parts:
            index = parts.index("trader")
            if index + 1 < len(parts):
                candidate = parts[index + 1]
                return candidate if re.fullmatch(r"[A-Za-z0-9_-]+", candidate) else None
        if parts and parts[-1].startswith("square-creator-"):
            return None
        if parts and parts[-1].isdigit():
            return parts[-1]
        return None
    if value.startswith("square-creator-"):
        return None
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else None


def _add_database(parser):
    parser.add_argument("--database", help="DuckDB path; defaults to config storage.path")


def _add_config(parser):
    parser.add_argument("--config", help="Optional TOML configuration file")


def _database_path(args):
    config = load_config(args.config)
    return args.database or config.storage.path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartmoney-research", description="Local research on publicly shared trading records")
    commands = parser.add_subparsers(dest="command", required=True)

    sync = commands.add_parser("sync", help="Fetch and store public trader data")
    _add_config(sync)
    _add_database(sync)
    sync.add_argument("--trader-id", help="Kopyon trader ID or a supported trader URL")
    sync.add_argument("--trader-name", help="Resolve a public trader by name when no Kopyon ID is supplied")
    sync.add_argument("--lookback-days", type=int, help="Override collection.days from config")
    sync.add_argument("--all-history", action="store_true", help="Request all history the provider returns")
    sync.add_argument("--dry-run", action="store_true", help="Print the planned date range without API calls")
    sync.add_argument("--now-ms", type=int, help=argparse.SUPPRESS)

    inspect = commands.add_parser("inspect", help="Report local data coverage and integrity")
    _add_config(inspect)
    _add_database(inspect)
    inspect.add_argument("--trader-id", required=True)

    analyze = commands.add_parser("analyze", help="Summarize collected trading behavior")
    _add_config(analyze)
    _add_database(analyze)
    analyze.add_argument("--trader-id", required=True)
    analyze.add_argument("--interval", default="1h")

    backtest = commands.add_parser("backtest", help="Run a local strategy backtest")
    _add_config(backtest)
    _add_database(backtest)
    backtest.add_argument("--trader-id", required=True)
    backtest.add_argument("--symbol", action="append", help="Repeat for each locally stored symbol; defaults to observed trader symbols")
    backtest.add_argument("--interval", default="1h")
    backtest.add_argument("--rule", choices=("ema_crossover", "rsi_threshold"), default="ema_crossover")
    backtest.add_argument("--direction", choices=("long", "short"), default="long")
    backtest.add_argument("--fast", type=int, default=12)
    backtest.add_argument("--slow", type=int, default=26)
    backtest.add_argument("--rsi-period", type=int, default=14)
    backtest.add_argument("--oversold", type=float, default=30)
    backtest.add_argument("--overbought", type=float, default=70)
    backtest.add_argument("--exit-level", type=float, default=50)
    backtest.add_argument("--parameter-grid", help='Finite JSON grid, for example {"fast":[8,12],"slow":[26]}')
    backtest.add_argument("--train-rows", type=int, default=240)
    backtest.add_argument("--test-rows", type=int, default=120)
    backtest.add_argument("--initial-cash", type=float, default=10000)
    backtest.add_argument("--leverage", type=float, default=1)
    backtest.add_argument("--max-leverage", type=float, default=2)
    backtest.add_argument("--position-size", type=float, default=.95)
    backtest.add_argument("--fee-rate", type=float, default=.0004)
    backtest.add_argument("--slippage-rate", type=float, default=.0002)
    backtest.add_argument("--stop-loss-pct", type=float)
    backtest.add_argument("--take-profit-pct", type=float)

    skill = commands.add_parser("skill", help="Generate a local evidence-cited strategy Skill")
    _add_config(skill)
    _add_database(skill)
    skill.add_argument("--trader-id", required=True)
    skill.add_argument("--output", default="generated_skills")

    export = commands.add_parser("export", help="Export a local database table")
    _add_config(export)
    _add_database(export)
    export.add_argument("--table", required=True)
    export.add_argument("--format", required=True, choices=("csv", "json", "parquet"))
    export.add_argument("--path", required=True)

    dashboard = commands.add_parser("dashboard", help="Open the read-only local dashboard")
    _add_config(dashboard)
    _add_database(dashboard)
    return parser


def _run_dashboard(args):
    from .dashboard import run_dashboard
    run_dashboard(args.database)


def _resolve_trader(api, trader_id: str | None, trader_name: str | None):
    parsed_id = parse_trader_id(trader_id)
    if parsed_id:
        return {"trader_id": parsed_id, "name": trader_name}
    if not trader_name:
        raise ValueError("Provide a Kopyon --trader-id or --trader-name for public name search")
    matches = api.search_trader(trader_name)
    exact = [row for row in matches if row.get("name", "").casefold() == trader_name.casefold()]
    candidates = exact or matches
    if len(candidates) != 1:
        raise ValueError(f"Trader search returned {len(candidates)} candidates; choose a unique name or provide --trader-id")
    match = candidates[0]
    resolved_id = match.get("traderId", match.get("trader_id", match.get("id")))
    if resolved_id is None:
        raise ValueError("Trader search result has no trader ID")
    return {"trader_id": str(resolved_id), "name": match.get("name", trader_name)}


def _json_safe(value):
    """Keep local analysis output machine-readable, including warm-up nulls."""
    import numpy as np
    import pandas as pd
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict('records'))
    if isinstance(value, pd.Series):
        return _json_safe([{'time': index, 'value': item} for index, item in value.items()])
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if value is pd.NA or value is pd.NaT or isinstance(value, float) and not math.isfinite(value):
        return None
    return value.isoformat() if hasattr(value, 'isoformat') else value


def _json_print(value):
    print(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False))


def _backtest_local(store, args):
    from .backtest import BacktestConfig, run_backtest, run_sensitivity, run_walk_forward
    from .dashboard import local_bars
    from .strategies import StrategyRule
    rule = StrategyRule(kind=args.rule, direction=args.direction, fast=args.fast, slow=args.slow,
                        rsi_period=args.rsi_period, oversold=args.oversold,
                        overbought=args.overbought, exit_level=args.exit_level)
    config = BacktestConfig(initial_cash=args.initial_cash, leverage=args.leverage,
                            max_leverage=args.max_leverage, position_size=args.position_size,
                            fee_rate=args.fee_rate, slippage_rate=args.slippage_rate,
                            stop_loss_pct=args.stop_loss_pct, take_profit_pct=args.take_profit_pct)
    if args.train_rows <= 0 or args.test_rows <= 0:
        raise ValueError('train-rows and test-rows must be positive')
    grid = json.loads(args.parameter_grid) if args.parameter_grid else (
        {'fast': [rule.fast], 'slow': [rule.slow]} if rule.kind == 'ema_crossover'
        else {'rsi_period': [rule.rsi_period], 'oversold': [rule.oversold]})
    symbols = args.symbol or sorted({row['symbol'] for table in ('position_history', 'open_positions', 'order_history')
                                    for row in store.rows(table) if row['trader_id'] == args.trader_id and row['symbol']})
    if not symbols:
        raise ValueError('No observed trader symbols. Supply --symbol for a local market-data experiment')
    funding = store.connection.execute('SELECT * FROM funding_rates ORDER BY funding_time').fetchdf()
    benchmark = local_bars(store, 'BTCUSDT', args.interval)
    market_coverage = {'BTCUSDT': benchmark.attrs['coverage']}
    benchmark = None if benchmark.empty else benchmark
    results = {}
    for symbol in dict.fromkeys(symbols):
        bars = local_bars(store, symbol, args.interval)
        market_coverage[symbol] = bars.attrs['coverage']
        if bars.empty:
            raise ValueError(f'No verified final {args.interval} bars for {symbol}; unavailable coverage: {market_coverage[symbol]}; run sync separately')
        if len(bars) <= args.train_rows:
            raise ValueError(f'{symbol} needs more than {args.train_rows} bars for chronological walk-forward; lower --train-rows or collect more data')
        results[symbol] = {
            'single': run_backtest(bars, funding, rule, config, benchmark_ohlcv=benchmark),
            'sensitivity': run_sensitivity(bars, funding, rule, grid, config, benchmark_ohlcv=benchmark),
            'walk_forward': run_walk_forward(bars, funding, rule, config, train_rows=args.train_rows,
                                             test_rows=args.test_rows, parameter_grid=grid,
                                             benchmark_ohlcv=benchmark),
        }
    return {'trader_id': args.trader_id, 'rule': rule.to_dict(), 'interval': args.interval,
            'label': 'derived', 'scope': 'allow-listed market-rule experiment; not a reconstruction of trader intent',
            'symbols': results, 'market_bar_coverage': market_coverage}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dashboard":
            args.database = _database_path(args)
            _run_dashboard(args)
            return 0
        if args.command == "sync":
            config = load_config(args.config)
            now_ms = args.now_ms if args.now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
            lookback_days = args.lookback_days if args.lookback_days is not None else config.collection.days
            if lookback_days <= 0:
                raise ValueError("lookback-days must be positive")
            start_ms = None if args.all_history else now_ms - lookback_days * 24 * 60 * 60 * 1000
            parsed_id = parse_trader_id(args.trader_id)
            if args.dry_run:
                if not parsed_id and not args.trader_name:
                    raise ValueError("Dry run needs --trader-id or --trader-name")
                _json_print({"dry_run": True, "trader_id": parsed_id, "trader_name": args.trader_name,
                             "start_ms": start_ms, "end_ms": now_ms, "market_type": config.collection.market_type,
                             "history_resources": ["positionHistory", "orderHistory"],
                             "market_symbols": list(config.collection.symbols),
                             "market_intervals": list(config.collection.intervals),
                             "market_data": ["klines", "fundingRate"]})
                return 0
            from .clients.kopyon import KopyonClient
            http = config.http
            api = KopyonClient(api_key=config.kopyon_key, timeout=http.timeout, min_interval=http.min_interval,
                               max_retries=http.max_retries, backoff_factor=http.backoff_factor)
            trader = _resolve_trader(api, args.trader_id, args.trader_name)
            database = args.database or config.storage.path
            with ResearchStore(database) as store:
                store.initialize()
                report = sync_trader(config, trader, store, lookback_days=lookback_days,
                                     all_history=args.all_history, kopyon_client=api)
            _json_print(report.to_dict())
            return 0 if report.complete else 1
        if args.command == "inspect":
            trader_id = parse_trader_id(args.trader_id)
            if not trader_id:
                raise ValueError("Provide a Kopyon trader ID")
            with ResearchStore(_database_path(args)) as store:
                store.initialize()
                _json_print(check_integrity(store, trader_id))
            return 0
        if args.command == "export":
            with ResearchStore(_database_path(args)) as store:
                store.initialize()
                output = export_table(store, args.table, args.format, args.path)
            print(str(output))
            return 0
        if args.command == "skill":
            from .analytics.evidence import build_evidence_bundle
            from .llm import LLMClient, generate_hypothesis, write_skill
            trader_id = parse_trader_id(args.trader_id)
            if not trader_id:
                raise ValueError("Provide a Kopyon trader ID")
            with ResearchStore(_database_path(args)) as store:
                store.initialize()
                bundle = build_evidence_bundle(store, trader_id)
            hypothesis = generate_hypothesis(bundle, LLMClient())
            output = write_skill(args.output, hypothesis, bundle)
            _json_print({"output": str(output), "skill": str(output / "SKILL.md"),
                         "evidence": str(output / "evidence.json")})
            return 0
        if args.command in {"analyze", "backtest"}:
            from .dashboard import analyze_local
            args.trader_id = parse_trader_id(args.trader_id)
            if not args.trader_id:
                raise ValueError("Provide a Kopyon trader ID")
            with ResearchStore(_database_path(args), read_only=True) as store:
                result = (analyze_local(store, args.trader_id, interval=args.interval)
                          if args.command == "analyze" else _backtest_local(store, args))
            _json_print(result)
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
