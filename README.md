smartmoney-research collects public trader records into a local DuckDB database, builds evidence-linked summaries, and tests allow-listed market rules with VectorBT. It reads Kopyon's documented public v5 routes and Binance public USDⓈ-M market data. It contains no exchange order execution.

Use Python 3.11 or later. Create a virtual environment and install the package with synthetic test support. The optional dashboard needs its own extra.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m smartmoney_research --help
# Optional browser dashboard
python -m pip install -e '.[dashboard]'
```

Copy `config.example.toml` to `config.toml` and `.env.example` to `.env`. The program reads credentials from the process environment and does not automatically load `.env`. Keep `.env` private and export it in your shell before collection. Obtain `KOPYON_KEY` from the [Kopyon console](https://console.kopyon.com/); a limited unauthenticated allowance is described in the [API reference](https://kopyon.com/docs/).

```sh
cp config.example.toml config.toml
cp .env.example .env
# Edit .env locally, then export its values
set -a
. ./.env
set +a
mkdir -p data reports outputs
```

The default database is `data/research.duckdb`. `--database` overrides `[storage].path`, and `--config` selects a TOML file. Configure HTTP timeout, throttling, retry/backoff, collection days, market type, overlap, symbols, and intervals using the example. Trader collection supports `UM` and `CM`; market bars and funding currently come only from the USDⓈ-M routes, so COIN-M market coverage is unavailable.

Use your provider trader ID in place of `TRADER_ID`. An explicit trader URL with an ID is accepted; a Binance Square creator slug is not a Kopyon trader ID. `--trader-name` resolves a unique public name through documented search. A dry run makes no network calls.

```sh
python -m smartmoney_research sync --config config.toml --trader-id TRADER_ID --dry-run
python -m smartmoney_research sync --config config.toml --trader-id TRADER_ID
python -m smartmoney_research inspect --config config.toml --trader-id TRADER_ID
```

The default history request covers 365 days. `--lookback-days 90` overrides that window. Explicit `--all-history` follows all history the provider returns, which can be shorter than the account's lifetime. Later runs resume from persisted watermarks with a configurable overlap, deduplicate normalized rows, refresh profile/open-position snapshots, and collect configured and observed symbols. A failed resource keeps its last successful checkpoint. Review sync reports and `inspect` for hidden history, missing costs, timestamp errors, and incomplete coverage.

```sh
python -m smartmoney_research sync --config config.toml --trader-id TRADER_ID --all-history
# Run this same command later for an incremental update
python -m smartmoney_research sync --config config.toml --trader-id TRADER_ID
```

For scheduled collection, run that incremental command through your own scheduler with absolute paths to the virtual-environment interpreter and configuration. Set the working directory and export credentials in the job environment. Run one writer at a time, retain the exit status and local log, and avoid simultaneous writer and reader processes on the same database. The dashboard closes its read connection after each snapshot. Choose the frequency to fit provider limits and your sampling needs; scheduling cannot reconstruct events the provider never exposes.

Analysis opens an existing database read-only and never fetches bars. It returns an evidence bundle with record IDs, source coverage, missing fields, metrics and per-event market context. Each event uses the final bar whose close time is strictly earlier than the event timestamp. Analysis and backtests require a known retrieval time strictly after bar close, and the close must be in the past. Rows with missing retrieval or close times, bars retrieved before closing, and future closes are excluded. Their counts remain visible as unavailable market-bar coverage in analysis, backtest reports and the dashboard, with reason counts that can overlap. Legacy rows lacking retrieval timestamps must be refreshed by sync before use. Missing market series, absent earlier bars, and indicator warm-up remain explicit missing values in the JSON.

```sh
python -m smartmoney_research analyze --config config.toml --trader-id TRADER_ID --interval 1h > reports/analysis.json
```

The indicator defaults are RSI 14 with Wilder-style exponential gain/loss smoothing, EMA 12 and 26, MACD as their difference with signal EMA 9, ATR 14 as a simple mean of true range, and volume mean/std/z-score over 20 bars. Trend strength is the absolute fast/slow EMA spread divided by ATR. Rolling volatility is the sample standard deviation of simple close returns over 20 bars, without annualization. Indicators are derived market features and do not establish why a trader acted.

Backtests consume only stored bars and funding. Choose `ema_crossover` or `rsi_threshold`, long or short, and a finite JSON parameter grid. Symbols default to those observed for the selected trader; repeat `--symbol` to choose local market experiments explicitly. Provide enough rows for training and at least one later test row. Parameter selection sees training rows only; every attempted grid point also receives an out-of-sample result. The overall sensitivity grid describes the whole sample and must not be used to claim independent validation.

```sh
python -m smartmoney_research backtest --config config.toml --trader-id TRADER_ID \
  --symbol ETHUSDT --interval 1h --rule ema_crossover --direction long \
  --parameter-grid '{"fast":[8,12],"slow":[26,32]}' \
  --train-rows 240 --test-rows 120 > reports/backtest.json
```

Default execution is 1x with a configurable ceiling no higher than 2x. The simulation reserves 5 percent of collateral for costs, charges 0.04 percent per fill, and assumes 0.02 percent slippage. Flags expose cash, position size, fees, slippage, stop loss and take profit. Close-derived signals execute at the next bar's open. OHLC stop checks start after the entry bar, with stop loss taking priority when both thresholds are crossed; OHLC cannot establish intrabar event order.

Funding is scoped to each symbol and to open positions at its timestamp. Missing mark prices and ambiguous funding inside a stop-exit bar are reported by the engine rather than treated as known costs. Missing funding rows mean incomplete funding coverage, even when the recorded funding adjustment is zero. Liquidation risk is an estimate using a maintenance-margin assumption; borrowing interest and full exchange liquidation mechanics are excluded. Return, drawdown, Sharpe, Sortino, profit factor and win rate come from simulated cost-adjusted equity. The BTC buy-and-hold baseline requires local BTCUSDT bars for every strategy timestamp and is gross, unlevered, and excludes costs. The report states when the benchmark is unavailable.

Export normalized tables or provenance for private analysis. Exports can contain trader records, so keep them in ignored local directories.

```sh
python -m smartmoney_research export --config config.toml --table position_history --format csv --path outputs/positions.csv
python -m smartmoney_research export --config config.toml --table order_history --format json --path outputs/orders.json
python -m smartmoney_research export --config config.toml --table klines --format parquet --path outputs/klines.parquet
```

The explicit `skill` command sends a local evidence bundle to an OpenAI-compatible model endpoint. Set `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` in the environment; the default base URL and model are shown in `.env.example`. Review what the bundle contains before selecting a remote endpoint. Missing credentials stop the command before an HTTP request. Evidence references, sample counts, allow-listed rules and risk limits are validated locally before writing `SKILL.md` and `evidence.json` under the ignored output directory.

```sh
python -m smartmoney_research skill --config config.toml --trader-id TRADER_ID --output generated_skills
python -m smartmoney_research dashboard --config config.toml
```

The dashboard binds to localhost and disables Streamlit usage telemetry. Its data helper opens DuckDB read-only and displays profile snapshots, coverage diagnostics, positions, separate sampled chart series, and strict as-of event indicators. Upload locally generated backtest JSON reports to compare rules, costs, funding, benchmarks and walk-forward results. Rendering makes no source API or LLM calls. It handles missing or empty databases without creating them; run sync separately to collect data.

Raw responses retain source, endpoint, parameters, retrieval time, response hash and payload. Normalized records reference that provenance, and unavailable values stay null. Profile chart PnL samples are not an account equity ledger; absent account equity means observed account drawdown cannot be calculated. Incomplete sharing, unstable identifiers, fingerprint deduplication, provider retention and changing snapshots limit inference. A tested market rule remains a hypothesis about observed behavior, not a reconstructed private trading strategy.

The MIT license covers this repository's code. The [data-source audit](docs/data-source-audit.md) documents source fields and licensing separately. [Kopyon's API terms](https://kopyon.com/binance-smart-money-api/terms/) restrict redistribution of substantially unchanged responses; this repository applies a stricter code-only policy and excludes downloaded records, personal reports, raw payloads and trader-specific generated Skills. Keep all such outputs local. The repository includes only generic documentation, schemas, prompts, code and synthetic test fixtures.
