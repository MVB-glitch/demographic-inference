from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demographic_inference.querychat_export import build_querychat_payload  # noqa: E402


DEFAULT_INPUT_JSON = ROOT / "results.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "querychat"

RESULTS_FILENAME = "inference_results.csv"
METRICS_FILENAME = "metric_summary.csv"
CONFUSION_FILENAME = "confusion_matrix.csv"
DISTRIBUTIONS_FILENAME = "distributions.csv"
STRUCTURED_JSON_FILENAME = "structured_inference_output.json"
METADATA_FILENAME = "metadata.json"
DUCKDB_FILENAME = "demographic_inference.duckdb"

STRING_COLUMNS = {
    "user_id",
    "truth_location_raw",
    "truth_location_group",
    "pred_location_raw",
    "pred_location_group",
    "truth_age_raw",
    "truth_age_group",
    "pred_age_raw",
    "pred_age_group",
    "truth_gender_raw",
    "truth_gender_group",
    "pred_gender_raw",
    "pred_gender_group",
    "combined_reasoning",
    "dimension",
    "metric",
    "truth",
    "prediction",
    "distribution",
    "label",
}


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare demographic inference JSON outputs for QueryChat.",
    )
    parser.add_argument(
        "input_json",
        nargs="?",
        default=DEFAULT_INPUT_JSON,
        help="Pipeline JSON output. Defaults to results.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated QueryChat files. Defaults to outputs/querychat.",
    )
    return parser.parse_args()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    for column in frame.columns:
        frame[column] = frame[column].map(_csv_value)

    for column in STRING_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].astype("string")

    return frame


def _flatten_distributions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, values in payload.get("distributions", {}).items():
        if not isinstance(values, list):
            continue
        for row in values:
            rows.append(
                {
                    "distribution": name,
                    "label": row.get("label"),
                    "count": row.get("count"),
                }
            )
    return rows


def _load_payload(input_json: Path) -> dict[str, Any]:
    with input_json.open("r", encoding="utf-8-sig") as source:
        data = json.load(source)

    if isinstance(data, dict) and "results" in data and "metrics" in data:
        return data

    if isinstance(data, list):
        return build_querychat_payload(data, source_file=input_json)

    raise ValueError(
        "Input JSON must be either the structured QueryChat export or the "
        "older list-of-records pipeline JSON."
    )


def _write_duckdb(tables: dict[str, pd.DataFrame], duckdb_path: Path) -> None:
    try:
        import duckdb
    except ImportError:
        print("duckdb is not installed; skipping DuckDB export.")
        return

    with duckdb.connect(str(duckdb_path)) as conn:
        for table_name, frame in tables.items():
            view_name = f"{table_name}_df"
            conn.register(view_name, frame)
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view_name}")
            conn.unregister(view_name)


def main() -> None:
    args = _parse_args()
    input_json = _resolve_path(args.input_json)
    output_dir = _resolve_path(args.output_dir)

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_payload(input_json)

    results = _records_to_frame(payload.get("results", []))
    metric_rows = _records_to_frame(payload.get("metric_rows", []))
    confusion_rows = _records_to_frame(payload.get("confusion_rows", []))
    distribution_rows = _records_to_frame(_flatten_distributions(payload))
    metadata = payload.get("metadata", {})

    results_path = output_dir / RESULTS_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    confusion_path = output_dir / CONFUSION_FILENAME
    distributions_path = output_dir / DISTRIBUTIONS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME
    structured_path = output_dir / STRUCTURED_JSON_FILENAME
    duckdb_path = output_dir / DUCKDB_FILENAME

    results.to_csv(results_path, index=False, encoding="utf-8")
    metric_rows.to_csv(metrics_path, index=False, encoding="utf-8")
    confusion_rows.to_csv(confusion_path, index=False, encoding="utf-8")
    distribution_rows.to_csv(distributions_path, index=False, encoding="utf-8")

    with metadata_path.open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)

    with structured_path.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)

    _write_duckdb(
        {
            "inference_results": results,
            "metric_summary": metric_rows,
            "confusion_matrix": confusion_rows,
            "distributions": distribution_rows,
        },
        duckdb_path,
    )

    print(f"Prepared QueryChat files in: {output_dir}")
    print(f"- {RESULTS_FILENAME}: {len(results):,} rows")
    print(f"- {METRICS_FILENAME}: {len(metric_rows):,} rows")
    print(f"- {CONFUSION_FILENAME}: {len(confusion_rows):,} rows")


if __name__ == "__main__":
    main()
