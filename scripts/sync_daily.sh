#!/usr/bin/env bash
set -euo pipefail

: "${KOPYON_KEY:?Set KOPYON_KEY in the process environment}"
: "${TRADER_ID:?Set TRADER_ID in the process environment}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATABASE="${RESEARCH_DATABASE:-data/research.duckdb}"
exec "$PYTHON_BIN" -m smartmoney_research sync --trader-id "$TRADER_ID" --database "$DATABASE"
