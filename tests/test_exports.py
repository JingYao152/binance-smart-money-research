import csv
import json

from smartmoney_research.exports import export_table
from smartmoney_research.storage import ResearchStore


def test_csv_and_json_exports_preserve_nulls_and_source_ids(tmp_path):
    with ResearchStore(tmp_path / "exports.duckdb") as store:
        store.initialize()
        store.upsert_history("position_history", [{
            "trader_id": "synthetic", "market_type": "UM", "record_id": "position-1", "id_kind": "provider",
            "symbol": "SYNUSDT", "net_pnl": None, "source": "kopyon", "raw_response_id": "raw-id-1",
        }])
        csv_path = tmp_path / "positions.csv"
        json_path = tmp_path / "positions.json"
        export_table(store, "position_history", "csv", csv_path)
        export_table(store, "position_history", "json", json_path)

        with csv_path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        exported = json.loads(json_path.read_text(encoding="utf-8"))[0]

    assert row["raw_response_id"] == "raw-id-1"
    assert row["net_pnl"] == "\\N"
    assert exported["raw_response_id"] == "raw-id-1"
    assert exported["net_pnl"] is None


def test_export_rejects_unknown_table_and_format(tmp_path):
    with ResearchStore(tmp_path / "exports.duckdb") as store:
        store.initialize()
        try:
            export_table(store, "unknown", "json", tmp_path / "out.json")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown table was accepted")
        try:
            export_table(store, "position_history", "xml", tmp_path / "out.xml")
        except ValueError:
            pass
        else:
            raise AssertionError("unknown format was accepted")


def test_parquet_export_reads_back_same_source_id_without_materializing_rows(tmp_path, monkeypatch):
    with ResearchStore(tmp_path / "exports.duckdb") as store:
        store.initialize()
        store.upsert_history("position_history", [{
            "trader_id": "synthetic", "market_type": "UM", "record_id": "position-1", "id_kind": "provider",
            "source": "kopyon", "raw_response_id": "raw-id-1",
        }])
        parquet_path = tmp_path / "positions.parquet"
        monkeypatch.setattr(store, "rows", lambda table: (_ for _ in ()).throw(AssertionError("Parquet must validate schema without fetching all rows")))
        export_table(store, "position_history", "parquet", parquet_path)
        assert parquet_path.exists()
        assert store.connection.execute("SELECT raw_response_id FROM read_parquet(?)", [str(parquet_path)]).fetchone() == ("raw-id-1",)
