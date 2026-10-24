# Binance Smart Money Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a public, code-only Python project that locally collects public Binance Smart Money data, analyzes it with market context, backtests low-leverage hypotheses, and generates evidence-cited local strategy Skills.

**Architecture:** A local CLI coordinates documented Kopyon v5 and Binance public API clients. DuckDB stores raw response envelopes, normalized rows, sync state, market data, and derived evidence. Analytics, VectorBT, an OpenAI-compatible LLM adapter, and a read-only Streamlit dashboard consume the same local database.

**Tech Stack:** Python 3.12+, DuckDB, pandas, NumPy, VectorBT, Streamlit, pytest, urllib from the standard library.

**Spec:** docs/superpowers/specs/2026-10-24-binance-smart-money-research-design.md

## Global Constraints

- Use only documented public Kopyon v5 and Binance market-data routes.
- Read KOPYON_KEY and LLM credentials from environment variables; never commit secrets.
- Default trader-history window is 365 days; all-history is an explicit option and means all history the provider returns.
- Record raw payload provenance and retain unavailable fields as missing.
- Do not commit downloaded records, reports, or trader-specific generated Skills.
- Trade-event features use only fully closed market bars available before the event.
- Backtests default to 1x leverage and enforce a configurable maximum of 2x.
- Unit tests use synthetic fixtures and run without network credentials.

---

### Task 1: Package scaffolding and documented API clients

**Files:**
- Create: pyproject.toml
- Create: .gitignore
- Create: .env.example
- Create: config.example.toml
- Create: src/smartmoney_research/__init__.py
- Create: src/smartmoney_research/config.py
- Create: src/smartmoney_research/clients/http.py
- Create: src/smartmoney_research/clients/kopyon.py
- Create: src/smartmoney_research/clients/binance.py
- Test: tests/test_config.py
- Test: tests/test_kopyon_client.py
- Test: tests/test_binance_client.py

**Interfaces:**
- KopyonClient.search_trader(name: str) returns matching trader dictionaries.
- KopyonClient.get_profile(trader_id: str) returns the provider data object.
- KopyonClient.get_chart(trader_id: str, period: str, chart_type: str) returns timestamp/value points.
- KopyonClient.get_open_positions(trader_id: str, market_type: str) returns a data object with positions, total, and hidden.
- KopyonClient.iter_history(resource: str, trader_id: str, market_type: str, start_ms: int | None, end_ms: int | None, rows: int) yields history rows while following nextCursor.
- BinanceClient.iter_klines(symbol: str, interval: str, start_ms: int, end_ms: int) and iter_funding_rates(symbol: str, start_ms: int, end_ms: int) yield normalized dictionaries.
- Public clients accept an injected transport so tests do not make network calls.

- [ ] **Step 1: Write tests before implementation**

Add a synthetic Kopyon response with two cursor pages. Assert the second request sends the first response’s nextCursor and the iterator yields both records once. Add a 429 then 200 transport sequence and assert the retry succeeds after a retry delay is requested. Assert missing KOPYON_KEY omits the auth header. Add Binance tests that parse one 12-field kline and advance startTime after the last returned bar.

- [ ] **Step 2: Run the focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_config.py tests/test_kopyon_client.py tests/test_binance_client.py -q
Expected: collection fails because smartmoney_research modules have not been created.

- [ ] **Step 3: Implement configuration and the HTTP transport**

Create typed configuration loading from TOML and environment variables. Add request timeout, conservative minimum interval, bounded retries, Retry-After support, and exponential backoff for 429 and 5xx. Do not retry non-retryable 4xx responses.

- [ ] **Step 4: Implement Kopyon v5 and Binance public clients**

Use the documented routes and query names. Detect a repeated cursor and raise a pagination error. Parse Kopyon error envelopes without dropping the status code. For Binance history, cap each request at the documented limit and advance from the last returned timestamp.

- [ ] **Step 5: Run focused tests and commit**

Run: python -m pytest tests/test_config.py tests/test_kopyon_client.py tests/test_binance_client.py -q
Expected: all focused tests pass without network access.

Commit message: feat: add public data clients

### Task 2: DuckDB evidence store, normalization, and integrity checks

**Files:**
- Create: src/smartmoney_research/storage.py
- Create: src/smartmoney_research/normalize.py
- Create: src/smartmoney_research/quality.py
- Test: tests/test_storage.py
- Test: tests/test_normalize.py
- Test: tests/test_quality.py

**Interfaces:**
- ResearchStore(path).initialize() creates schema version, raw_responses, traders, profiles, charts, open_positions, position_history, order_history, klines, funding_rates, and sync_state tables.
- ResearchStore.save_raw(source: str, endpoint: str, params: dict, payload: object, status: int, fetched_at: datetime) stores a stable response hash and the complete local JSON payload. Payload accepts any JSON value because Binance K-line responses are arrays.
- ResearchStore.upsert_history(table: str, records: list[dict]) inserts or updates rows idempotently.
- normalize_position(trader_id: str, market_type: str, record: dict) and normalize_order(...) preserve source_details and raw fields.
- check_integrity(store, trader_id: str) returns counts, hidden-history flags, missing-field counts, duplicate counts, cursor errors, timestamp errors, and PnL reconciliation residuals.

- [ ] **Step 1: Write tests before implementation**

Test that the same position ID upserted twice produces one row and that two raw responses with different retrieval times remain separate evidence records. Test order fingerprint stability for records without provider IDs. Test that a hidden history flag is reported as incomplete, missing fees remain null, and net PnL reconciliation reports a mismatch without changing source values.

- [ ] **Step 2: Run focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_storage.py tests/test_normalize.py tests/test_quality.py -q
Expected: collection fails because storage and normalization modules do not exist yet.

- [ ] **Step 3: Implement DuckDB schema and provenance storage**

Use parameterized SQL. Store source snapshots separately from current normalized state. Use provider positionId when present; otherwise create and persist a documented SHA-256 fingerprint. Store missing source fields as SQL NULL.

- [ ] **Step 4: Implement normalization and quality reporting**

Keep every normalized row linked to the source trader, market, raw response ID, and fetch timestamp. Calculate reconciliation residual as source netPnl minus grossPnl minus signed fundingPnl plus positive fee fields. Report residuals only.

- [ ] **Step 5: Run focused tests and commit**

Run: python -m pytest tests/test_storage.py tests/test_normalize.py tests/test_quality.py -q
Expected: all focused tests pass.

Commit message: feat: store auditable trader data

### Task 3: Resumable ingestion, exports, and scheduled sync

**Files:**
- Create: src/smartmoney_research/ingest.py
- Create: src/smartmoney_research/exports.py
- Create: src/smartmoney_research/cli.py
- Create: src/smartmoney_research/__main__.py
- Create: scripts/sync_daily.sh
- Create: deploy/crontab.example
- Test: tests/test_ingest.py
- Test: tests/test_exports.py
- Test: tests/test_cli.py

**Interfaces:**
- sync_trader(config, trader, store, lookback_days: int = 365, all_history: bool = False) returns a SyncReport.
- export_table(store, table: str, format: str, path: Path) writes CSV, JSON, or Parquet from local DuckDB rows.
- CLI commands are sync, inspect, analyze, backtest, skill, export, and dashboard.

- [ ] **Step 1: Write tests before implementation**

Use fake API clients and a temporary DuckDB file. Assert default sync sets a 365-day start timestamp, all-history omits the lower bound, cursor pages are saved once, a second sync is idempotent, an overlap window is used for incremental updates, and a cursor cycle fails visibly. Assert CSV and JSON exports preserve nulls and source IDs.

- [ ] **Step 2: Run focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_ingest.py tests/test_exports.py tests/test_cli.py -q
Expected: collection fails because ingestion, exports, and CLI modules are not created yet.

- [ ] **Step 3: Implement ingestion and durable checkpoints**

Save each response before normalization. Maintain per-trader, market, resource high-water timestamps. Use a configurable overlap and provider cursor pagination. Record sync status and error details without advancing the checkpoint on incomplete runs.

- [ ] **Step 4: Implement exports and CLI**

Export only when explicitly requested. Add a dry-run option to show planned calls and time range. The dashboard command reads local data and never triggers a sync implicitly. Add shell and cron examples that call the CLI with KOPYON_KEY provided in the process environment.

- [ ] **Step 5: Run focused tests and commit**

Run: python -m pytest tests/test_ingest.py tests/test_exports.py tests/test_cli.py -q
Expected: all focused tests pass.

Commit message: feat: add resumable sync and exports

### Task 4: Market features and trader behavior analytics

**Files:**
- Create: src/smartmoney_research/analytics/features.py
- Create: src/smartmoney_research/analytics/metrics.py
- Create: src/smartmoney_research/analytics/evidence.py
- Test: tests/test_features.py
- Test: tests/test_metrics.py
- Test: tests/test_evidence.py

**Interfaces:**
- compute_indicators(ohlcv: DataFrame, config: FeatureConfig) returns a DataFrame with RSI, EMA fast/slow, MACD line/signal/histogram, ATR, volume statistics, trend strength, and rolling volatility.
- align_trade_events(events: DataFrame, bars: DataFrame) returns events joined only to the latest fully closed bar before each event timestamp.
- analyze_positions(positions: DataFrame, chart_points: DataFrame | None) returns count, win rate, payoff ratio, profit factor, realized PnL, funding, fees, holding time, long/short split, leverage distribution, and drawdown where a valid equity series exists.
- build_evidence_bundle(store, trader_id: str) returns claim candidates with evidence IDs, sample counts, coverage, and missing-field notes.

- [ ] **Step 1: Write tests before implementation**

Use small synthetic OHLCV and trade fixtures. Assert the aligned bar close_time is strictly earlier than the trade timestamp. Check known EMA/RSI/ATR values within tolerance. Verify zero-loss data does not produce infinite profit factor, partial positions are labeled, and absent equity data yields a missing drawdown rather than a fabricated value.

- [ ] **Step 2: Run focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_features.py tests/test_metrics.py tests/test_evidence.py -q
Expected: collection fails because analytics modules do not exist yet.

- [ ] **Step 3: Implement indicators and point-in-time alignment**

Use pandas and NumPy formulas with warm-up rows represented as missing. Join on candle close_time with backward as-of semantics and reject events with no prior completed bar.

- [ ] **Step 4: Implement behavioral metrics and evidence references**

Use closed-position records for realized performance. Treat open-position unrealized PnL separately. Treat Kopyon chart points as sampled source metrics and never imply they are transaction-level equity.

- [ ] **Step 5: Run focused tests and commit**

Run: python -m pytest tests/test_features.py tests/test_metrics.py tests/test_evidence.py -q
Expected: all focused tests pass.

Commit message: feat: add point-in-time trading analytics

### Task 5: VectorBT backtests with explicit costs and chronological validation

**Files:**
- Create: src/smartmoney_research/backtest.py
- Create: src/smartmoney_research/strategies.py
- Test: tests/test_strategies.py
- Test: tests/test_backtest.py

**Interfaces:**
- StrategyRule supports allow-listed EMA crossover and RSI threshold rules only in version one.
- run_backtest(ohlcv: DataFrame, funding: DataFrame, rule: StrategyRule, config: BacktestConfig) returns trade records, equity series, assumptions, and metrics.
- run_walk_forward(...) returns chronological train/test windows and out-of-sample metrics.
- run_sensitivity(...) evaluates the configured finite parameter grid without selecting from the test window.

- [ ] **Step 1: Write tests before implementation**

Create deterministic bars where a signal occurs at one close and assert execution is at the next bar open. Assert configured trading fees and slippage reduce value, funding is charged to longs and credited to shorts for matching signed rates, the 1x default and 2x ceiling are enforced, and walk-forward test dates always follow training dates.

- [ ] **Step 2: Run focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_strategies.py tests/test_backtest.py -q
Expected: collection fails because strategy and backtest modules do not exist yet.

- [ ] **Step 3: Implement allow-listed signal rules and VectorBT execution**

Shift close-derived signals by one bar and pass next-bar open as execution price. Pass configured fees and slippage to VectorBT. Use OHLC high/low stop inputs for configured stop loss and take profit; calculate a separate conservative liquidation estimate from leverage and configured maintenance-margin assumptions.

- [ ] **Step 4: Apply funding and calculate reported metrics**

Apply each signed funding rate only to a position open at fundingTime, using the Binance funding response markPrice when available. Report the number of funding rows used and any missing-price assumptions. Calculate return, Sharpe, Sortino, profit factor, win rate, and maximum drawdown from the cost-adjusted result. Compare with a BTC buy-and-hold benchmark over the same timestamps.

- [ ] **Step 5: Add walk-forward and parameter sensitivity**

Split chronologically into configured train and test windows. Parameter selection reads training data only. Report every attempted parameter set and its out-of-sample result.

- [ ] **Step 6: Run focused tests and commit**

Run: python -m pytest tests/test_strategies.py tests/test_backtest.py -q
Expected: all focused tests pass.

Commit message: feat: add cost-aware vectorbt backtests

### Task 6: LLM evidence contract and local strategy Skill generation

**Files:**
- Create: src/smartmoney_research/llm.py
- Create: prompts/strategy_research.md
- Create: skills/trader-research/SKILL.md
- Create: schemas/strategy-hypothesis.schema.json
- Test: tests/test_llm.py
- Test: tests/test_skill_generation.py

**Interfaces:**
- validate_hypothesis(payload: dict, evidence_ids: set[str]) returns a validated StrategyHypothesis or raises a typed validation error.
- generate_hypothesis(bundle: EvidenceBundle, client: LLMClient) returns a validated StrategyHypothesis.
- render_skill(hypothesis: StrategyHypothesis, bundle: EvidenceBundle) returns UTF-8 Markdown containing tested conditions, risk rules, sample size, evidence IDs, and missing-data notes.
- CLI writes trader-specific SKILL.md and evidence.json to a local output directory ignored by Git.

- [ ] **Step 1: Write tests before implementation**

Use a fake LLM response. Assert unknown evidence IDs, unsupported strategy rule names, missing sample counts, and statements of intent without evidence are rejected. Assert generated Markdown contains every cited ID and explicitly includes the observed/derived/hypothesis labels. Assert no HTTP request is made without LLM_API_KEY.

- [ ] **Step 2: Run focused tests and confirm the expected missing-module failures**

Run: python -m pytest tests/test_llm.py tests/test_skill_generation.py -q
Expected: collection fails because LLM and Skill modules do not exist yet.

- [ ] **Step 3: Implement strict schema and prompt**

Keep the model response within the versioned JSON schema. Evidence IDs must be a subset of the supplied bundle. The prompt forbids invented trades, missing-field substitution, trader-intent claims, and unsupported causal language.

- [ ] **Step 4: Implement local Markdown rendering**

Generate only a research Skill, never exchange orders. Include source coverage, sample count, strategy conditions, sizing, leverage cap, scale-in rules, exits, costs, drawdown controls, validation windows, and evidence IDs.

- [ ] **Step 5: Run focused tests and commit**

Run: python -m pytest tests/test_llm.py tests/test_skill_generation.py -q
Expected: all focused tests pass.

Commit message: feat: generate evidence-cited strategy skills

### Task 7: Dashboard, documentation, CI, and release audit

**Files:**
- Create: src/smartmoney_research/dashboard.py
- Modify: src/smartmoney_research/cli.py
- Modify: .env.example
- Create: README.md
- Create: LICENSE
- Create: .github/workflows/tests.yml
- Create: docs/data-source-audit.md
- Test: tests/test_dashboard_data.py
- Modify: tests/test_cli.py

- [ ] **Step 1: Write dashboard-data and local CLI workflow tests**

Assert that dashboard queries read only local DuckDB, gracefully show empty tables, and expose hidden-history and missing-field coverage. No live API call may occur while rendering the dashboard. Add synthetic DuckDB CLI tests: `analyze` returns an evidence bundle with per-event point-in-time indicators, while `backtest` consumes locally stored bars and funding, runs an allow-listed rule, and returns a chronological walk-forward and sensitivity result without writing to the database or making network calls.

- [ ] **Step 2: Run the dashboard-data test and confirm the expected missing-module failure**

Run: python -m pytest tests/test_dashboard_data.py -q
Expected: collection fails because the dashboard module does not exist yet.

- [ ] **Step 3: Implement the read-only dashboard and wire the research CLI**

Display profile snapshots, source coverage, positions, sampled PnL chart, per-trade indicators, and backtest result comparisons. Label estimated and derived values in the UI. Wire `analyze` to the local evidence bundle and market-context alignment. Wire `backtest` to local DuckDB K-lines and funding rates, with allow-listed rule selection, cost-aware single runs, training-only walk-forward parameter selection, sensitivity results, and BTC baseline comparison. Set `bars.attrs["symbol"]` before funding and benchmark matching. Neither command may fetch data or mutate stored source records. Correct `.env.example` to match `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.

- [ ] **Step 4: Write setup, provenance, limitation, and scheduling documentation**

Document install, KOPYON_KEY setup, config, 365-day sync, all-history sync, incremental runs, database location, CSV/JSON/Parquet exports, indicator definitions, backtest assumptions, local Skill outputs, and the no-redistribution boundary. The data-source audit distinguishes verified live schema from fields that remain unavailable.

- [ ] **Step 5: Add CI and complete verification**

Run: python -m pytest -q
Expected: all synthetic tests pass without network credentials.

Run: python -m smartmoney_research --help
Expected: all commands are listed and exit status is zero.

Commit message: docs: add dashboard and public release guidance

- [ ] **Step 6: Review public contents and create the code-only repository**

Check Git status for .env, DuckDB, exports, raw responses, reports, and generated Skills. Confirm none are tracked. Create the public repository binance-smart-money-research under the authenticated GitHub account, push the reviewed code, and verify the remote README and default branch.
