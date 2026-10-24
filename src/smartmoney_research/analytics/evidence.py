"""Evidence bundles that keep each derived claim linked to stored source rows."""
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json

import pandas as pd

from ..quality import check_integrity
from .metrics import analyze_positions

_RAW_ENDPOINTS = {
    "position_history": "/v5/binance-smart-money/positionHistory",
    "open_positions": "/v5/binance-smart-money/openPositions",
    "order_history": "/v5/binance-smart-money/orderHistory",
    "profiles": "/v5/binance-smart-money/traderProfile",
    "traders": "/v5/binance-smart-money/traderProfile",
    "charts": "/v5/binance-smart-money/traderChart",
}
_RAW_MARKET_SCOPED_TABLES = frozenset({"position_history", "open_positions", "order_history"})
_RAW_ENDPOINT_SET = frozenset(_RAW_ENDPOINTS.values())


@dataclass(frozen=True)
class EvidenceBundle:
    trader_id: str
    coverage_status: str
    incomplete: bool
    coverage: dict
    sample_counts: dict
    missing_fields: dict
    metrics: dict
    evidence_ids: set[str]
    items: dict[str, dict]
    claim_candidates: list[dict]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["evidence_ids"] = sorted(self.evidence_ids)
        return result


def _json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _rows(store, table: str, trader_id: str) -> list[dict]:
    cursor = store.connection.execute(f"SELECT * FROM {table} WHERE trader_id = ?", [trader_id])
    names = [column[0] for column in cursor.description]
    result = []
    for values in cursor.fetchall():
        row = dict(zip(names, values))
        for field in ("raw_fields", "source_details"):
            if field in row:
                row[field] = _json_value(row[field])
        result.append(row)
    return result


def _display_value(value):
    return value.isoformat() if isinstance(value, (datetime, date)) else value


def _row_evidence_id(table: str, row: dict) -> str:
    if "record_id" in row:
        return f"{table}:{row.get('market_type') or 'unknown'}:{row['record_id']}"
    identity = row.get("snapshot_id") or row.get("trader_id")
    return f"{table}:{identity}"


def _item(table: str, row: dict, evidence_id: str) -> dict:
    item = {
        "evidence_id": evidence_id,
        "label": "observed",
        "table": table,
        "record_id": row.get("record_id") or row.get("snapshot_id") or row.get("trader_id"),
        "market_type": row.get("market_type"),
        "source": row.get("source"),
        "raw_response_id": row.get("raw_response_id"),
        "fetched_at": _display_value(row.get("fetched_at")),
        "observed_fields": row.get("raw_fields") or {},
    }
    return item


def _raw_response_item(
    store,
    raw_id: str,
    trader_id: str,
    expected_endpoint: str | None,
    expected_market_type: str | None = None,
    require_market_type: bool = False,
):
    raw = store.connection.execute(
        "SELECT source, endpoint, params, fetched_at, content_hash, status FROM raw_responses WHERE raw_response_id = ?",
        [raw_id],
    ).fetchone()
    if raw is None:
        return None
    params = _json_value(raw[2])
    if isinstance(params, dict):
        scoped_trader_id = params.get("traderId") or params.get("trader_id")
        scoped_market_type = params.get("marketType") or params.get("market_type")
    else:
        scoped_trader_id = None
        scoped_market_type = None
    if raw[0] != "kopyon" or str(scoped_trader_id) != trader_id or raw[1] != expected_endpoint:
        return None
    if require_market_type and (
        expected_market_type is None
        or scoped_market_type is None
        or str(scoped_market_type).upper() != str(expected_market_type).upper()
    ):
        return None
    return {
        "evidence_id": f"raw:{raw_id}", "label": "observed", "table": "raw_responses",
        "raw_response_id": raw_id, "source": raw[0], "endpoint": raw[1],
        "params": params, "fetched_at": _display_value(raw[3]),
        "content_hash": raw[4], "status": int(raw[5]),
    }


def _known_row_ids(rows: list[dict], field: str) -> list[str]:
    return [row["evidence_id"] for row in rows if row.get(field) is not None]


def _candidate(claim: str, value, rows: list[dict], sample_count: int, *, missing_fields=None) -> dict:
    return {
        "claim": claim,
        "label": "derived",
        "value": value,
        "sample_count": int(sample_count),
        "evidence_ids": [row["evidence_id"] for row in rows],
        "missing_fields": list(missing_fields or []),
    }


def build_evidence_bundle(store, trader_id: str) -> EvidenceBundle:
    """Build a compact summary whose claim candidates point to source records.

    Provider chart values remain observed sampled metrics. Because a PNL chart
    is not an account-equity ledger, it is not used to invent drawdown or equity.
    """
    trader_id = str(trader_id)
    tables = ("position_history", "open_positions", "order_history", "profiles", "charts", "traders")
    rows_by_table = {table: _rows(store, table, trader_id) for table in tables}
    item_map: dict[str, dict] = {}
    evidence_ids: set[str] = set()
    tagged_rows: dict[str, list[dict]] = {}
    raw_response_ids = set()
    dangling_raw_references = []
    dangling_raw_ids = set()

    for table, rows in rows_by_table.items():
        tagged = []
        for source_row in rows:
            evidence_id = _row_evidence_id(table, source_row)
            evidence_ids.add(evidence_id)
            tagged.append({**source_row, "evidence_id": evidence_id})
            raw_id = source_row.get("raw_response_id")
            if raw_id:
                raw_id = str(raw_id)
                requires_market = table in _RAW_MARKET_SCOPED_TABLES
                raw_item = _raw_response_item(
                    store,
                    raw_id,
                    trader_id,
                    _RAW_ENDPOINTS.get(table),
                    expected_market_type=source_row.get("market_type") if requires_market else None,
                    require_market_type=requires_market,
                )
                if raw_item:
                    raw_response_ids.add(raw_id)
                    raw_evidence_id = f"raw:{raw_id}"
                    evidence_ids.add(raw_evidence_id)
                    item_map[raw_evidence_id] = raw_item
                    raw_status = "verified"
                else:
                    dangling_raw_ids.add(raw_id)
                    dangling_raw_references.append({"evidence_id": evidence_id, "raw_response_id": raw_id})
                    raw_status = "unresolved"
            else:
                raw_status = "missing"
            row_item = _item(table, source_row, evidence_id)
            row_item["raw_response_status"] = raw_status
            item_map[evidence_id] = row_item
        tagged_rows[table] = tagged

    raw_rows = store.connection.execute(
        """SELECT raw_response_id, source, endpoint, params, fetched_at, content_hash, status
           FROM raw_responses
           WHERE source = 'kopyon'
             AND (json_extract_string(params, '$.traderId') = ?
                  OR json_extract_string(params, '$.trader_id') = ?)""",
        [trader_id, trader_id],
    ).fetchall()
    for raw in raw_rows:
        raw_id = str(raw[0])
        if raw_id in dangling_raw_ids:
            continue
        if raw[2] not in _RAW_ENDPOINT_SET:
            continue
        raw_response_ids.add(raw_id)
        raw_evidence_id = f"raw:{raw_id}"
        evidence_ids.add(raw_evidence_id)
        item_map[raw_evidence_id] = {
            "evidence_id": raw_evidence_id, "label": "observed", "table": "raw_responses",
            "raw_response_id": raw_id, "source": raw[1], "endpoint": raw[2],
            "params": _json_value(raw[3]), "fetched_at": _display_value(raw[4]),
            "content_hash": raw[5], "status": int(raw[6]),
        }

    history = rows_by_table["position_history"]
    open_positions = rows_by_table["open_positions"]
    combined = pd.DataFrame(history + open_positions)
    chart_points = pd.DataFrame(rows_by_table["charts"])
    metrics = analyze_positions(combined, chart_points if not chart_points.empty else None)
    integrity = check_integrity(store, trader_id)
    coverage = {table: len(rows) for table, rows in rows_by_table.items()}
    coverage["raw_responses"] = len(raw_response_ids)
    coverage["diagnostics"] = {
        "hidden_history_count": len(integrity["hidden_history"]),
        "cursor_error_count": len(integrity["cursor_errors"]),
        "timestamp_error_count": len(integrity["timestamp_errors"]),
        "sync_error_count": len(integrity["sync_errors"]),
        "pnl_unverifiable_count": integrity["pnl_unverifiable_count"],
        "missing_raw_response_count": len(dangling_raw_ids),
        "missing_raw_response_reference_count": len(dangling_raw_references),
    }

    observed_symbols = sorted({str(row["symbol"]) for table in ("position_history", "open_positions", "order_history")
                               for row in rows_by_table[table] if row.get("symbol")})
    market_counts = {}
    for symbol in observed_symbols:
        market_counts[symbol] = {
            "klines": store.connection.execute("SELECT COUNT(*) FROM klines WHERE symbol = ?", [symbol]).fetchone()[0],
            "funding_rates": store.connection.execute("SELECT COUNT(*) FROM funding_rates WHERE symbol = ?", [symbol]).fetchone()[0],
        }
    coverage["market_data_by_symbol"] = market_counts

    missing_fields = {
        table: dict(counts)
        for table, counts in integrity["missing_fields"].items()
    }
    if dangling_raw_references:
        missing_fields["provenance"] = {
            "raw_response_id_unresolved": len(dangling_raw_references),
            "dangling_raw_response_ids": sorted(dangling_raw_ids),
        }
    sample_counts = {
        "position_history": len(history),
        "open_positions": len(open_positions),
        "order_history": len(rows_by_table["order_history"]),
        "closed_positions": metrics["closed_position_count"],
        "known_pnl_outcomes": metrics["pnl_sample_count"],
    }

    candidates = []
    closed_statuses = {"CLOSED", "CLOSE", "FINISHED", "LIQUIDATED", "LIQUIDATION"}
    closed_known = [row for row in tagged_rows["position_history"] + tagged_rows["open_positions"]
                    if (row.get("close_time") is not None
                        or str(row.get("status") or "").upper() in closed_statuses)
                    and row.get("net_pnl") is not None]
    if metrics["pnl_sample_count"]:
        candidates.append(_candidate("known_outcome_win_rate", metrics["win_rate"], closed_known,
                                     metrics["pnl_sample_count"],
                                     missing_fields=["net_pnl"] if metrics["pnl_missing_count"] else []))
        candidates.append(_candidate("known_realized_net_pnl", metrics["realized_pnl"], closed_known,
                                     metrics["pnl_sample_count"],
                                     missing_fields=["net_pnl"] if not metrics["realized_pnl_complete"] else []))
    position_rows = tagged_rows["position_history"] + tagged_rows["open_positions"]
    for symbol, summary in metrics["per_symbol"].items():
        symbol_rows = [row for row in position_rows if str(row.get("symbol") or "").strip().upper() == symbol]
        symbol_missing = list(summary["missing_fields"])
        if any(str(row.get("raw_response_id")) in dangling_raw_ids for row in symbol_rows):
            symbol_missing.append("raw_response_id")
        symbol_missing = sorted(set(symbol_missing))
        candidate = _candidate("symbol_position_outcomes", summary, symbol_rows,
                               summary["pnl_sample_count"], missing_fields=symbol_missing)
        candidate["symbol"] = symbol
        candidates.append(candidate)
    leverage_rows = [row for row in tagged_rows["position_history"] + tagged_rows["open_positions"]
                     if row.get("leverage") is not None]
    if leverage_rows:
        candidates.append(_candidate("observed_leverage_distribution", metrics["leverage_distribution"],
                                     leverage_rows, len(leverage_rows)))
    direction_rows = [row for row in tagged_rows["position_history"] + tagged_rows["open_positions"]
                      if row.get("side") is not None]
    if direction_rows:
        candidates.append(_candidate("observed_direction_counts", metrics["side_counts"],
                                     direction_rows, len(direction_rows)))
    partial_ids = set(metrics["partial_position_ids"])
    partial_rows = [row for row in tagged_rows["position_history"] + tagged_rows["open_positions"]
                    if str(row.get("record_id") or row.get("position_id")) in partial_ids]
    if partial_rows:
        candidates.append(_candidate("observed_partial_closes", metrics["partial_close_count"],
                                     partial_rows, len(partial_rows)))

    if integrity["coverage_status"] != "uncollected":
        source_rows = [row for table in tables for row in tagged_rows[table]]
        if source_rows:
            source_missing = []
            if integrity["incomplete"]:
                source_missing.append("trader history availability")
            if dangling_raw_references:
                source_missing.append("raw_response_id")
            candidates.append(_candidate("source_record_coverage", coverage, source_rows,
                                         len(source_rows),
                                         missing_fields=source_missing))

    bundle_missing = dict(missing_fields)
    bundle_missing["analysis"] = list(metrics["missing_fields"])
    if metrics["drawdown_status"] != "observed_equity_series":
        bundle_missing["analysis"].append("explicit account equity series unavailable; sampled PNL chart is not treated as equity")

    return EvidenceBundle(
        trader_id=trader_id,
        coverage_status=integrity["coverage_status"],
        incomplete=integrity["incomplete"] or bool(dangling_raw_references),
        coverage=coverage,
        sample_counts=sample_counts,
        missing_fields=bundle_missing,
        metrics=metrics,
        evidence_ids=evidence_ids,
        items=item_map,
        claim_candidates=candidates,
    )
