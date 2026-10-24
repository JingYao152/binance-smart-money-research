# Binance Smart Money Research Design

## Goal

Build a local-first Python research system that collects only publicly shared Binance Smart Money trader data, preserves source evidence, studies behavior with market context, and produces testable low-leverage strategy hypotheses. The first research profile is selected locally by the user; public source code contains no trader-specific identity, collected records, or generated personal report.

## Existing project survey

Tavily searches of GitHub found useful partial projects, but no repository matching the complete requested workflow.

- Alorse/copy-trading-intel is an MIT-licensed multi-exchange copy-trading audit project. It includes resumable scrapers, analyses, and a SKILL.md endpoint reference. Its Binance source is the copy-trading lead-portfolio product, and its skill is a reference and analysis workflow rather than a per-trader, evidence-validated strategy Skill generator. See https://github.com/Alorse/copy-trading-intel and https://github.com/Alorse/copy-trading-intel/blob/master/SKILL.md.
- ramayzzz/Binance-Copy-Trading provides lead-portfolio analytics, charts, and LLM-ready JSON exports. The repository description does not show an evidence-checked LLM strategy generator or an out-of-sample low-leverage backtest pipeline. See https://github.com/ramayzzz/Binance-Copy-Trading.
- 0xBennie/binance-smart-money-oi-monitor exposes coin-level Binance Smart Money and market signals to MCP clients. It monitors aggregate positioning by symbol rather than one named trader’s historical account. See https://github.com/0xBennie/binance-smart-money-oi-monitor.
- Nunnito/Binance-Futures-Leaderboard-API documents the Kopyon service and its trader/history routes. It supplies the data API rather than the local audit, research, backtest, and generated Skill layers. See https://github.com/Nunnito/Binance-Futures-Leaderboard-API.

The new project will use Kopyon’s documented API instead of scraping private Binance endpoints or relying on browser requests that require access workarounds. The supported API currently uses the v5 route family under https://api.kopyon.com/v5/binance-smart-money. Search resolves a trader name to an API trader ID; profile, chart, open-position, position-history, and order-history routes expose data for a single trader. History is cursor-paginated. The live schema probe confirmed the route family is reachable and that the profile, positions, and histories contain the fields listed below. These live responses are implementation evidence only and are not included in the public repository.

## Data sources and limits

Kopyon is the source of the trader profile and trading records. Its profile includes performance metrics and sharing flags. Open positions include symbol, side, size, entry and mark prices, leverage, margin, unrealized PnL, and source-specific margin data. Position history includes position identifiers, status, side, leverage, entry and close prices, open and close times, opened and closed volume, gross PnL, funding PnL, trading fee, liquidation fee, and net PnL when costs are known. Order history includes symbol, order side, position side, price, quantity, quote quantity, and timestamps.

The public order records do not confirm a stable order ID, per-order commission, per-order realized PnL, or historical margin. Funding PnL is available at position level, while individual funding charge events are not confirmed. Profile charts are sampled series rather than a complete account ledger. Missing fields stay missing. Any values derived from public market bars must use the label derived and retain the source event IDs used.

Binance USDⓈ-M public market endpoints provide K-lines and funding-rate history. They can supply OHLCV and market funding features, but they do not grant access to another account’s private trades. Binance account trade history is a signed USER_DATA endpoint tied to the account owner’s key and has a limited history window. No private Binance credentials, login sessions, hidden API paths, or access-control workarounds are in scope.

Kopyon documentation describes cursor pagination and a free keyed monthly quota; the API reference also describes a small unauthenticated quota. The client will accept KOPYON_KEY from the environment, respect published limits, back off on 429/5xx errors, detect repeated cursors, and keep raw responses locally. Kopyon API terms restrict redistribution of substantially unchanged responses and separately allow derived outputs. This project applies a stricter code-only release policy. The GitHub repository therefore contains source code, synthetic fixtures, schemas, and prompts only. Downloaded records, analysis reports, and trader-specific generated Skill files are local outputs ignored by Git.

References:

- Kopyon Smart Money endpoints: https://kopyon.com/binance-smart-money-api/
- Kopyon API reference and paging conventions: https://kopyon.com/docs/
- Kopyon API terms: https://kopyon.com/binance-smart-money-api/terms/
- Separate Kopyon application terms: https://kopyon.com/terms/
- Binance USDⓈ-M market data: https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data
- Binance USDⓈ-M signed trade endpoint: https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade

## Architecture

The repository is a single Python package with narrow modules. A CLI owns scheduled and manual collection. A Kopyon client handles trader search, profile, chart, open positions, cursor-based histories, authentication, retries, and throttling. A Binance client downloads public K-lines and funding data. DuckDB stores raw response envelopes, normalized source records, sync checkpoints, market bars, and derived analysis. CSV, JSON, and Parquet exports are local.

The default collection window is the most recent 365 days. An explicit all-history option follows every available cursor without claiming that the provider retains every event since account creation. Incremental sync starts from a stored event watermark with a configurable overlap, then deduplicates records. Every raw response records the source, route, query parameters, retrieval time, response hash, and payload. Integrity reports identify hidden history, missing fields, cursor problems, duplicate keys, timestamp issues, and PnL reconciliation residuals.

Analysis joins each trade event to the last fully closed market bar available before the event. The feature set includes RSI, EMA, MACD, ATR, volume, trend strength, realized volatility, and funding context. Trade summaries report symbol and direction preference, leverage, position size, holding time, scale-in/out evidence, realized and unrealized results, fees, drawdown, payoff ratio, profit factor, and win rate. Observed source fields and derived values remain separate.

VectorBT evaluates allow-listed strategy rules from point-in-time features. Signals are shifted to the next bar for execution; fees, slippage, funding, and configurable maintenance-margin liquidation assumptions are explicit inputs. Validation includes chronological holdouts, rolling windows, parameter sensitivity, and a BTC buy-and-hold benchmark. The default simulated leverage is 1x, with a configurable hard cap of 2x. Liquidation calculations are labeled estimates because public historical position margin brackets are not supplied for every record.

The LLM module creates a compact evidence bundle and requests a JSON strategy hypothesis from an OpenAI-compatible endpoint configured through environment variables. It validates that cited evidence IDs exist and that the proposed rule uses the allow-listed backtest schema. It distinguishes observed facts, derived statistics, and hypotheses, and refuses claims about trader intent. A trader-specific SKILL.md is generated locally alongside its JSON evidence manifest.

A local Streamlit dashboard reads DuckDB and displays source coverage, profile snapshots, current positions, position history, equity/PNL chart points, indicators, risk statistics, and backtest comparisons. It does not submit orders.

## Data and evidence rules

- Source payloads are immutable local evidence. Normalized rows retain their source and retrieval timestamp.
- Position IDs are the preferred position-history keys. Orders without provider IDs use a documented content fingerprint and report possible collision risk.
- Missing fields are null or absent, never silently filled from price movement or LLM output.
- PnL reconciliation uses gross PnL plus signed funding PnL minus positive fees. A mismatch is reported, not overwritten.
- LLM claims cite evidence IDs and sample counts. Unsupported claims are rejected or labeled as hypotheses.
- Public code and synthetic tests never contain the live trader’s records or personal strategy output.

## Phased delivery

1. Source verification artifacts, configuration, Kopyon and Binance clients, throttling, retries, pagination, and local DuckDB storage.
2. Incremental synchronization, idempotent ingestion, data-integrity and coverage reports, exports, and a scheduled CLI example.
3. Point-in-time market alignment, indicator features, behavioral and risk statistics, and dashboard views.
4. VectorBT strategy rules, cost and funding treatment, conservative leverage limits, rolling out-of-sample validation, parameter sensitivity, and baseline comparison.
5. Evidence-bounded LLM prompt, JSON validation, local Skill generation, and reproducible reports.
6. User-facing documentation, synthetic tests, CI, MIT license, and public code-only GitHub release.

## Acceptance criteria

- A user can resolve a public trader name or provide an ID, configure KOPYON_KEY without editing source, and collect 365 days by default or all provider-accessible history on request.
- Pagination, retries, rate limiting, deduplication, raw-response provenance, and incremental checkpoints are deterministic and covered by tests using synthetic data.
- Missing or hidden data is visible in coverage output and never represented as zero.
- Binance K-lines and funding requests are public and use bounded pagination.
- Indicator values at a trade timestamp use only candles fully closed before that timestamp.
- Backtest results include assumptions and report return, Sharpe, Sortino, profit factor, maximum drawdown, fees, slippage, funding, and estimated liquidation risk.
- Generated strategy output contains only validated fields and references evidence present in the local database.
- The dashboard is read-only and the package does not include live provider data, credentials, reports, or generated trader-specific Skills.
- Unit and integration tests run locally without external API credentials.
