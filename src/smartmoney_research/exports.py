"""Explicit local exports from the allowlisted DuckDB evidence tables."""
import csv
from datetime import date, datetime
import json
from pathlib import Path


_NULL = r"\N"


def _columns(store, table: str):
    return [(name, str(type_name).upper()) for name, type_name in store.table_schema(table)]


def _json_value(value, type_name: str):
    if value is None:
        return None
    if type_name == "JSON" and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def export_table(store, table: str, format: str, path: Path) -> Path:
    """Write one local allowlisted table as CSV, JSON, or Parquet.

    CSV uses the explicit ``\\N`` token for SQL NULL so it remains distinguishable
    from an empty string. JSON uses its standard ``null`` representation.
    """
    output = Path(path)
    kind = format.lower()
    if kind not in ("csv", "json", "parquet"):
        raise ValueError("format must be csv, json, or parquet")
    columns = _columns(store, table)
    output.parent.mkdir(parents=True, exist_ok=True)

    if kind == "parquet":
        # The schema-only table allowlist check avoids materializing large tables.
        store.connection.execute(f"COPY (SELECT * FROM {table}) TO ? (FORMAT PARQUET)", [str(output)])
        return output

    rows = store.rows(table)
    names = [name for name, _ in columns]
    type_by_name = dict(columns)
    if kind == "json":
        serializable = [{name: _json_value(row.get(name), type_by_name[name]) for name in names} for row in rows]
        with output.open("w", encoding="utf-8", newline="") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
        return output

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            values = {}
            for name in names:
                value = row.get(name)
                if value is None:
                    values[name] = _NULL
                elif isinstance(value, (dict, list)):
                    values[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                elif isinstance(value, (datetime, date)):
                    values[name] = value.isoformat()
                else:
                    values[name] = value
            writer.writerow(values)
    return output


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON")
