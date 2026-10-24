from datetime import datetime, timezone

from smartmoney_research.analytics.evidence import EvidenceBundle, build_evidence_bundle
from smartmoney_research.normalize import normalize_position
from smartmoney_research.storage import ResearchStore


def test_evidence_bundle_uses_source_ids_and_reports_coverage_and_missing_fields(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ResearchStore(tmp_path / "evidence.duckdb") as store:
        store.initialize()
        raw_id = store.save_raw("kopyon", "/v5/binance-smart-money/positionHistory", {"traderId": "synthetic", "marketType": "UM"},
                                {"data": {"positions": [{"positionId": "p-1"}]}}, 200, now)
        store.upsert_history("position_history", [
            normalize_position("synthetic", "UM", {
                "positionId": "p-1", "symbol": "SYNUSDT", "side": "LONG", "status": "CLOSED",
                "openedAt": 1_000, "closedAt": 2_000, "leverage": 2, "netPnl": 10,
                "grossPnl": 11, "fundingPnl": 0, "tradingFee": 1, "liquidationFee": 0,
            }, raw_response_id=raw_id, fetched_at=now),
            normalize_position("synthetic", "UM", {
                "positionId": "p-2", "symbol": "SYNUSDT", "side": "SHORT", "status": "CLOSED",
                "openedAt": 3_000, "closedAt": 4_000, "leverage": 4, "netPnl": -5,
            }, raw_response_id=raw_id, fetched_at=now),
            normalize_position("synthetic", "UM", {
                "positionId": "p-3", "symbol": "SYNUSDT", "side": "LONG", "status": "CLOSED",
                "openedAt": 5_000, "leverage": 3, "netPnl": 20,
            }, raw_response_id=raw_id, fetched_at=now),
        ])

        bundle = build_evidence_bundle(store, "synthetic")

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.coverage["position_history"] == 3
    assert bundle.sample_counts["closed_positions"] == 3
    assert bundle.coverage_status == "observed"
    assert "position_history:UM:p-1" in bundle.evidence_ids
    assert "position_history:UM:p-2" in bundle.evidence_ids
    assert "position_history:UM:p-3" in bundle.evidence_ids
    assert bundle.items["position_history:UM:p-1"]["raw_response_id"] == raw_id
    assert bundle.items["position_history:UM:p-1"]["label"] == "observed"
    assert bundle.missing_fields["position_history"]["trading_fee"] == 2
    win_rate = next(item for item in bundle.claim_candidates if item["claim"] == "known_outcome_win_rate")
    assert win_rate["label"] == "derived"
    assert win_rate["sample_count"] == 3
    assert set(win_rate["evidence_ids"]) == {
        "position_history:UM:p-1", "position_history:UM:p-2", "position_history:UM:p-3"
    }


def test_uncollected_trader_has_no_claim_candidates_or_invented_zero_metrics(tmp_path):
    with ResearchStore(tmp_path / "empty.duckdb") as store:
        store.initialize()
        bundle = build_evidence_bundle(store, "unseen")

    assert bundle.coverage_status == "uncollected"
    assert bundle.sample_counts["position_history"] == 0
    assert bundle.claim_candidates == []
    assert bundle.metrics["realized_pnl"] is None
    assert bundle.metrics["win_rate"] is None


def test_empty_provider_response_is_counted_as_observed_evidence(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ResearchStore(tmp_path / "empty-observed.duckdb") as store:
        store.initialize()
        raw_id = store.save_raw("kopyon", "/v5/binance-smart-money/positionHistory", {"traderId": "empty-trader", "marketType": "UM"},
                                {"data": {"positions": [], "hidden": False}}, 200, now)
        bundle = build_evidence_bundle(store, "empty-trader")

    assert bundle.coverage_status == "observed"
    assert bundle.coverage["raw_responses"] == 1
    assert f"raw:{raw_id}" in bundle.evidence_ids
    assert bundle.items[f"raw:{raw_id}"]["endpoint"] == "/v5/binance-smart-money/positionHistory"
    assert bundle.claim_candidates == []


def test_evidence_coverage_explains_hidden_history_as_incomplete(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ResearchStore(tmp_path / "hidden.duckdb") as store:
        store.initialize()
        store.save_raw("kopyon", "/v5/binance-smart-money/positionHistory", {"traderId": "hidden-trader", "marketType": "UM"},
                       {"data": {"positions": [], "hidden": True}}, 200, now)
        bundle = build_evidence_bundle(store, "hidden-trader")

    assert bundle.incomplete is True
    assert bundle.coverage["diagnostics"]["hidden_history_count"] == 1
    assert bundle.claim_candidates == []


def test_per_symbol_claim_evidence_matches_only_that_symbols_source_rows(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ResearchStore(tmp_path / "symbols.duckdb") as store:
        store.initialize()
        raw_id = store.save_raw("kopyon", "/v5/binance-smart-money/positionHistory", {"traderId": "symbols", "marketType": "UM"},
                                {"data": {"positions": []}}, 200, now)
        records = [
            {"positionId": "btc-a", "symbol": "BTCUSDT", "status": "CLOSED", "closedAt": 2_000, "netPnl": 10},
            {"positionId": "btc-b", "symbol": "BTCUSDT", "status": "CLOSED", "closedAt": 3_000},
            {"positionId": "eth-a", "symbol": "ETHUSDT", "status": "CLOSED", "closedAt": 4_000, "netPnl": -5},
        ]
        store.upsert_history("position_history", [
            normalize_position("symbols", "UM", row, raw_response_id=raw_id, fetched_at=now)
            for row in records
        ])

        bundle = build_evidence_bundle(store, "symbols")

    bitcoin_claim = next(
        claim for claim in bundle.claim_candidates
        if claim["claim"] == "symbol_position_outcomes" and claim["symbol"] == "BTCUSDT"
    )
    assert set(bitcoin_claim["evidence_ids"]) == {
        "position_history:UM:btc-a", "position_history:UM:btc-b"
    }
    assert bitcoin_claim["sample_count"] == 1
    assert bitcoin_claim["value"]["position_count"] == 2
    assert bitcoin_claim["value"]["pnl_missing_count"] == 1
    assert bitcoin_claim["value"]["realized_pnl"] == 10.0
    assert bitcoin_claim["value"]["win_rate"] == 1.0


def test_dangling_raw_response_reference_is_not_valid_evidence_and_marks_bundle_incomplete(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with ResearchStore(tmp_path / "dangling.duckdb") as store:
        store.initialize()
        store.upsert_history("position_history", [normalize_position(
            "dangling", "UM", {
                "positionId": "row-with-missing-source", "symbol": "SYNUSDT", "status": "CLOSED",
                "side": "LONG", "openedAt": 1_000, "closedAt": 2_000,
                "netPnl": 1, "grossPnl": 1, "fundingPnl": 0,
                "tradingFee": 0, "liquidationFee": 0,
            }, raw_response_id="missing-raw-id", fetched_at=now,
        )])

        bundle = build_evidence_bundle(store, "dangling")

    assert "position_history:UM:row-with-missing-source" in bundle.evidence_ids
    assert "raw:missing-raw-id" not in bundle.evidence_ids
    assert bundle.incomplete is True
    assert bundle.coverage["diagnostics"]["missing_raw_response_count"] == 1
    assert bundle.missing_fields["provenance"]["raw_response_id_unresolved"] == 1
    coverage_claim = next(claim for claim in bundle.claim_candidates if claim["claim"] == "source_record_coverage")
    assert "raw_response_id" in coverage_claim["missing_fields"]
    symbol_claim = next(claim for claim in bundle.claim_candidates if claim["claim"] == "symbol_position_outcomes")
    assert "raw_response_id" in symbol_claim["missing_fields"]


def test_raw_evidence_must_match_trader_and_normalized_resource_route(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    position_route = "/v5/binance-smart-money/positionHistory"
    order_route = "/v5/binance-smart-money/orderHistory"
    with ResearchStore(tmp_path / "wrong-scope.duckdb") as store:
        store.initialize()
        other_trader_raw = store.save_raw("kopyon", position_route,
                                         {"traderId": "another-trader", "marketType": "UM"},
                                         {"data": {"positions": []}}, 200, now)
        wrong_route_raw = store.save_raw("kopyon", order_route,
                                        {"traderId": "scoped-trader", "marketType": "UM"},
                                        {"data": {"orders": []}}, 200, now)
        wrong_source_raw = store.save_raw("binance", position_route,
                                         {"traderId": "scoped-trader", "marketType": "UM"},
                                         {"data": {"positions": []}}, 200, now)
        wrong_market_raw = store.save_raw("kopyon", position_route,
                                         {"traderId": "scoped-trader", "marketType": "CM"},
                                         {"data": {"positions": []}}, 200, now)
        store.upsert_history("position_history", [
            normalize_position("scoped-trader", "UM", {
                "positionId": "wrong-owner", "symbol": "SYNUSDT", "side": "LONG", "status": "CLOSED",
                "openedAt": 1_000, "closedAt": 2_000, "netPnl": 1, "grossPnl": 1,
                "fundingPnl": 0, "tradingFee": 0, "liquidationFee": 0,
            }, raw_response_id=other_trader_raw, fetched_at=now),
            normalize_position("scoped-trader", "UM", {
                "positionId": "wrong-route", "symbol": "SYNUSDT", "side": "SHORT", "status": "CLOSED",
                "openedAt": 3_000, "closedAt": 4_000, "netPnl": -1, "grossPnl": -1,
                "fundingPnl": 0, "tradingFee": 0, "liquidationFee": 0,
            }, raw_response_id=wrong_route_raw, fetched_at=now),
            normalize_position("scoped-trader", "UM", {
                "positionId": "wrong-source", "symbol": "SYNUSDT", "side": "LONG", "status": "CLOSED",
                "openedAt": 5_000, "closedAt": 6_000, "netPnl": 2, "grossPnl": 2,
                "fundingPnl": 0, "tradingFee": 0, "liquidationFee": 0,
            }, raw_response_id=wrong_source_raw, fetched_at=now),
            normalize_position("scoped-trader", "UM", {
                "positionId": "wrong-market", "symbol": "SYNUSDT", "side": "SHORT", "status": "CLOSED",
                "openedAt": 7_000, "closedAt": 8_000, "netPnl": -2, "grossPnl": -2,
                "fundingPnl": 0, "tradingFee": 0, "liquidationFee": 0,
            }, raw_response_id=wrong_market_raw, fetched_at=now),
        ])

        bundle = build_evidence_bundle(store, "scoped-trader")

    assert "position_history:UM:wrong-owner" in bundle.evidence_ids
    assert "position_history:UM:wrong-route" in bundle.evidence_ids
    assert "position_history:UM:wrong-source" in bundle.evidence_ids
    assert "position_history:UM:wrong-market" in bundle.evidence_ids
    assert f"raw:{other_trader_raw}" not in bundle.evidence_ids
    assert f"raw:{wrong_route_raw}" not in bundle.evidence_ids
    assert f"raw:{wrong_source_raw}" not in bundle.evidence_ids
    assert f"raw:{wrong_market_raw}" not in bundle.evidence_ids
    assert bundle.incomplete is True
    assert bundle.coverage["raw_responses"] == 0
    assert bundle.coverage["diagnostics"]["missing_raw_response_reference_count"] == 4
    assert bundle.missing_fields["provenance"]["raw_response_id_unresolved"] == 4
    assert set(bundle.missing_fields["provenance"]["dangling_raw_response_ids"]) == {
        other_trader_raw, wrong_route_raw, wrong_source_raw, wrong_market_raw
    }
