from __future__ import annotations

import html
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from chatlas import ChatGoogle
from dotenv import load_dotenv
from querychat import QueryChat
from shiny import App, Inputs, Outputs, Session, reactive, render, ui


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "outputs" / "querychat"
RESULTS_PATH = DATA_DIR / "inference_results.csv"
METRICS_PATH = DATA_DIR / "metric_summary.csv"
CONFUSION_PATH = DATA_DIR / "confusion_matrix.csv"
METADATA_PATH = DATA_DIR / "metadata.json"
GREETING = Path(__file__).parent / "greeting.md"
DATA_DESCRIPTION = Path(__file__).parent / "data_description.md"
STYLES = Path(__file__).parent / "styles.css"

TEXT_COLUMNS = {
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
}

NUMERIC_COLUMNS = {
    "location_belief",
    "age_belief",
    "gender_belief",
    "dst_k",
    "dst_non_kanto",
    "dst_omega",
    "max_conflict",
    "posts_kanto_mass",
    "posts_non_kanto_mass",
    "posts_omega_mass",
    "posts_confidence",
    "network_kanto_mass",
    "network_non_kanto_mass",
    "network_omega_mass",
    "network_confidence",
    "bio_kanto_mass",
    "bio_non_kanto_mass",
    "bio_omega_mass",
    "bio_confidence",
    "images_kanto_mass",
    "images_non_kanto_mass",
    "images_omega_mass",
    "images_confidence",
    "age_mass_18_24",
    "age_mass_25_34",
    "age_mass_35_44",
    "age_mass_45_54",
    "age_mass_55_64",
    "age_mass_65_plus",
    "age_mass_omega",
    "gender_mass_male",
    "gender_mass_female",
    "gender_mass_omega",
    "age_abs_error_years",
    "correct_dimensions",
    "evaluated_dimensions",
}

BOOL_COLUMNS = {"location_correct", "age_correct", "gender_correct"}

DISPLAY_COLUMNS = [
    "user_id",
    "truth_location_group",
    "pred_location_group",
    "location_correct",
    "location_belief",
    "truth_age_group",
    "pred_age_group",
    "age_correct",
    "age_abs_error_years",
    "age_belief",
    "truth_gender_group",
    "pred_gender_group",
    "gender_correct",
    "gender_belief",
    "max_conflict",
    "dst_k",
    "dst_non_kanto",
    "dst_omega",
]

DIMENSIONS = {
    "location": {
        "label": "Location",
        "truth": "truth_location_group",
        "prediction": "pred_location_group",
        "correct": "location_correct",
        "labels": ["Kanto", "Non-Kanto"],
    },
    "age": {
        "label": "Age",
        "truth": "truth_age_group",
        "prediction": "pred_age_group",
        "correct": "age_correct",
        "labels": ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    },
    "gender": {
        "label": "Gender",
        "truth": "truth_gender_group",
        "prediction": "pred_gender_group",
        "correct": "gender_correct",
        "labels": ["Male", "Female"],
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as source:
        data = json.load(source)
    return data if isinstance(data, dict) else {}


def _querychat_tools() -> tuple[str, ...]:
    if os.getenv("QUERYCHAT_ENABLE_VIZ", "true").lower() in {"0", "false", "no"}:
        return ("update", "query")

    modules = ("altair", "ggsql", "shinywidgets", "vl_convert")
    if all(importlib.util.find_spec(module) is not None for module in modules):
        return ("update", "query", "visualize")

    return ("update", "query")


def _load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Missing QueryChat data. Run "
            "`python querychat_explorer/prepare_querychat_data.py results.json` first."
        )

    frame = pd.read_csv(RESULTS_PATH, dtype={column: "string" for column in TEXT_COLUMNS})
    return _clean_frame(frame)


def _load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        return pd.DataFrame(columns=["dimension", "metric", "value"])
    frame = pd.read_csv(METRICS_PATH, dtype={"dimension": "string", "metric": "string"})
    if "value" in frame:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame


def _load_confusion() -> pd.DataFrame:
    if not CONFUSION_PATH.exists():
        return pd.DataFrame(columns=["dimension", "truth", "prediction", "count"])
    frame = pd.read_csv(
        CONFUSION_PATH,
        dtype={"dimension": "string", "truth": "string", "prediction": "string"},
    )
    if "count" in frame:
        frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0).astype(int)
    return frame


def _as_pandas(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if hasattr(data, "collect"):
        collected = data.collect()
        if hasattr(collected, "to_pandas"):
            return collected.to_pandas()
        return pd.DataFrame(collected)
    return pd.DataFrame(data)


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in NUMERIC_COLUMNS.intersection(clean.columns):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in BOOL_COLUMNS.intersection(clean.columns):
        clean[column] = (
            clean[column].eq(True)
            | clean[column].astype("string").str.lower().isin({"true", "1", "yes"})
        )
    return clean


def _format_count(value: Any) -> str:
    numeric = 0 if pd.isna(value) else int(value)
    return f"{numeric:,}"


def _format_percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _empty_state(message: str) -> str:
    return f"<div class='empty-state'>{html.escape(message)}</div>"


def _mean_bool(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.mean())


def _classification_rows(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    config = DIMENSIONS[dimension]
    required = [config["truth"], config["prediction"]]
    if frame.empty or not set(required).issubset(frame.columns):
        return pd.DataFrame(columns=required)

    rows = frame.dropna(subset=required).copy()
    rows = rows[(rows[config["truth"]] != "") & (rows[config["prediction"]] != "")]
    return rows


def _dimension_metrics(frame: pd.DataFrame, dimension: str) -> dict[str, Any]:
    config = DIMENSIONS[dimension]
    rows = _classification_rows(frame, dimension)
    truth_count = 0
    if config["truth"] in frame:
        truth_count = int(frame[config["truth"]].dropna().astype("string").ne("").sum())
    correct = 0
    if not rows.empty:
        correct = int((rows[config["truth"]] == rows[config["prediction"]]).sum())

    labels = config["labels"]
    precision_values = []
    recall_values = []
    f1_values = []
    for label in labels:
        tp = int(((rows[config["truth"]] == label) & (rows[config["prediction"]] == label)).sum())
        fp = int(((rows[config["truth"]] != label) & (rows[config["prediction"]] == label)).sum())
        fn = int(((rows[config["truth"]] == label) & (rows[config["prediction"]] != label)).sum())
        support = int((rows[config["truth"]] == label).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else None
        )
        if support:
            if precision is not None:
                precision_values.append(precision)
            if recall is not None:
                recall_values.append(recall)
            if f1 is not None:
                f1_values.append(f1)

    metrics = {
        "truth_count": truth_count,
        "evaluable": len(rows),
        "correct": correct,
        "uncertain": max(0, truth_count - len(rows)),
        "accuracy": correct / len(rows) if len(rows) else None,
        "precision": sum(precision_values) / len(precision_values)
        if precision_values
        else None,
        "recall": sum(recall_values) / len(recall_values) if recall_values else None,
        "f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "coverage": len(rows) / truth_count if truth_count else None,
    }
    if dimension == "age" and "age_abs_error_years" in rows:
        mae = rows["age_abs_error_years"].dropna().mean()
        metrics["mae_years"] = None if pd.isna(mae) else float(mae)
    return metrics


def _metric_tiles(frame: pd.DataFrame) -> str:
    location = _dimension_metrics(frame, "location")
    age = _dimension_metrics(frame, "age")
    gender = _dimension_metrics(frame, "gender")
    conflict = frame["max_conflict"].mean() if "max_conflict" in frame else None

    tiles = [
        ("Users", _format_count(len(frame)), "Filtered rows", "blue"),
        ("Location Accuracy", _format_percent(location["accuracy"]), "Kanto vs Non-Kanto", "green"),
        ("Age Accuracy", _format_percent(age["accuracy"]), "Exact age bucket", "amber"),
        ("Gender Accuracy", _format_percent(gender["accuracy"]), "Male/Female labels", "violet"),
        ("Age MAE", _format_number(age.get("mae_years"), 2), "Years from bucket midpoint", "red"),
        ("Mean Conflict", _format_number(conflict), "DST location fusion", "teal"),
        (
            "Uncertain Predictions",
            _format_count(location["uncertain"] + age["uncertain"] + gender["uncertain"]),
            "Across evaluated dimensions",
            "slate",
        ),
    ]

    html_tiles = []
    for label, value, detail, color in tiles:
        html_tiles.append(
            f"<article class='metric-tile metric-{color}'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{html.escape(value)}</strong>"
            f"<small>{html.escape(detail)}</small>"
            "</article>"
        )
    return f"<div class='metric-grid'>{''.join(html_tiles)}</div>"


def _prediction_distribution(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return _empty_state("No rows in the current selection.")
    counts = frame[column].fillna("Uncertain/Missing").replace("", "Uncertain/Missing").value_counts()
    max_count = max(int(counts.max()), 1)
    rows = []
    for label, count in counts.items():
        width = max((int(count) / max_count) * 100, 4)
        safe_label = html.escape(str(label))
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label' title='{safe_label}'>{safe_label}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width:.2f}%'></div>"
            "</div>"
            f"<div class='bar-value'>{int(count):,}</div>"
            "</div>"
        )
    return f"<div class='bar-chart'>{''.join(rows)}</div>"


def _performance_table(frame: pd.DataFrame) -> str:
    rows = []
    for key, config in DIMENSIONS.items():
        metrics = _dimension_metrics(frame, key)
        rows.append(
            "<tr>"
            f"<th>{html.escape(config['label'])}</th>"
            f"<td>{_format_count(metrics['evaluable'])}</td>"
            f"<td>{_format_percent(metrics['coverage'])}</td>"
            f"<td>{_format_percent(metrics['accuracy'])}</td>"
            f"<td>{_format_percent(metrics['precision'])}</td>"
            f"<td>{_format_percent(metrics['recall'])}</td>"
            f"<td>{_format_percent(metrics['f1'])}</td>"
            f"<td>{_format_number(metrics.get('mae_years'), 2) if key == 'age' else ''}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table class='metric-table'>"
        "<thead><tr><th>Dimension</th><th>Eval</th><th>Coverage</th>"
        "<th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>MAE</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _confusion_matrix(frame: pd.DataFrame, dimension: str) -> str:
    config = DIMENSIONS[dimension]
    rows = _classification_rows(frame, dimension)
    labels = config["labels"]
    if rows.empty:
        return _empty_state("No evaluable rows in the current selection.")

    counts = (
        rows.groupby([config["truth"], config["prediction"]])
        .size()
        .reset_index(name="count")
    )
    lookup = {
        (row[config["truth"]], row[config["prediction"]]): int(row["count"])
        for _, row in counts.iterrows()
    }
    max_count = max(lookup.values()) if lookup else 1

    header = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    body_rows = []
    for truth in labels:
        cells = []
        for prediction in labels:
            count = lookup.get((truth, prediction), 0)
            alpha = 0.08 + 0.62 * (count / max_count if max_count else 0)
            cls = "is-match" if truth == prediction else "is-miss"
            cells.append(
                f"<td class='{cls}' style='--cell-alpha:{alpha:.3f}'>"
                f"{count:,}</td>"
            )
        body_rows.append(
            f"<tr><th>{html.escape(truth)}</th>{''.join(cells)}</tr>"
        )

    return (
        "<div class='confusion-wrap'>"
        f"<h4>{html.escape(config['label'])}</h4>"
        "<table class='confusion-table'>"
        f"<thead><tr><th>Truth \\ Prediction</th>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _belief_chart(frame: pd.DataFrame) -> str:
    belief_columns = [
        ("Location", "location_belief", "#2c7be5"),
        ("Age", "age_belief", "#c98117"),
        ("Gender", "gender_belief", "#6d5bd0"),
    ]
    rows = []
    for label, column, color in belief_columns:
        if column not in frame or frame.empty:
            value = None
        else:
            value = frame[column].dropna().mean()
        width = 0 if value is None or pd.isna(value) else max(float(value) * 100, 2)
        rows.append(
            "<div class='bar-row belief-row'>"
            f"<div class='bar-label'>{html.escape(label)}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{width:.2f}%; background:{color}'></div>"
            "</div>"
            f"<div class='bar-value'>{_format_number(value)}</div>"
            "</div>"
        )
    return f"<div class='bar-chart'>{''.join(rows)}</div>"


def _dst_evidence_chart(frame: pd.DataFrame) -> str:
    sources = [
        ("Posts", "posts_confidence", "posts_kanto_mass", "posts_non_kanto_mass", "posts_omega_mass"),
        (
            "Network",
            "network_confidence",
            "network_kanto_mass",
            "network_non_kanto_mass",
            "network_omega_mass",
        ),
        ("Bio", "bio_confidence", "bio_kanto_mass", "bio_non_kanto_mass", "bio_omega_mass"),
        (
            "Images",
            "images_confidence",
            "images_kanto_mass",
            "images_non_kanto_mass",
            "images_omega_mass",
        ),
    ]
    if frame.empty:
        return _empty_state("No rows in the current selection.")

    def _stack_width(value: Any) -> float:
        if value is None or pd.isna(value):
            return 0.0
        return max(float(value) * 100, 0.0)

    rows = []
    for label, conf_col, k_col, nk_col, omega_col in sources:
        conf = frame[conf_col].dropna().mean() if conf_col in frame else None
        k = frame[k_col].dropna().mean() if k_col in frame else 0
        nk = frame[nk_col].dropna().mean() if nk_col in frame else 0
        omega = frame[omega_col].dropna().mean() if omega_col in frame else 0
        rows.append(
            "<div class='evidence-row'>"
            f"<div class='evidence-label'><strong>{html.escape(label)}</strong>"
            f"<span>conf {_format_number(conf)}</span></div>"
            "<div class='stacked-bar'>"
            f"<span class='stack-k' style='width:{_stack_width(k):.2f}%'></span>"
            f"<span class='stack-nk' style='width:{_stack_width(nk):.2f}%'></span>"
            f"<span class='stack-omega' style='width:{_stack_width(omega):.2f}%'></span>"
            "</div></div>"
        )

    return (
        "<div class='evidence-chart'>"
        f"{''.join(rows)}"
        "<div class='stack-legend'>"
        "<span><i class='stack-k'></i>Kanto</span>"
        "<span><i class='stack-nk'></i>Non-Kanto</span>"
        "<span><i class='stack-omega'></i>Omega</span>"
        "</div></div>"
    )


def _reasoning_cards(frame: pd.DataFrame) -> str:
    if frame.empty:
        return _empty_state("No users in the current selection.")

    selected = frame.copy()
    if "max_conflict" in selected:
        selected = selected.sort_values("max_conflict", ascending=False, na_position="last")

    cards = []
    for _, row in selected.head(5).iterrows():
        user_id = html.escape(str(row.get("user_id", "")))
        location = html.escape(str(row.get("pred_location_group") or "Uncertain"))
        age = html.escape(str(row.get("pred_age_group") or "Uncertain"))
        gender = html.escape(str(row.get("pred_gender_group") or "Uncertain"))
        conflict = _format_number(row.get("max_conflict"))
        reasoning = html.escape(str(row.get("combined_reasoning") or "No reasoning available."))
        cards.append(
            "<article class='reasoning-card'>"
            "<div class='reasoning-meta'>"
            f"<strong>{user_id}</strong><span>Conflict {conflict}</span>"
            "</div>"
            f"<div class='prediction-line'>{location} / {age} / {gender}</div>"
            f"<p>{reasoning[:900]}</p>"
            "</article>"
        )
    return f"<div class='reasoning-grid'>{''.join(cards)}</div>"


def _pipeline_summary(metadata: dict[str, Any], metrics: pd.DataFrame, confusion: pd.DataFrame) -> str:
    total_users = metadata.get("total_users", 0)
    generated_at = metadata.get("generated_at") or "Unknown"
    schema_version = metadata.get("schema_version") or "Unknown"
    kanto_prefectures = metadata.get("kanto_prefectures") or []
    metric_count = len(metrics)
    confusion_count = int(confusion["count"].sum()) if "count" in confusion else 0

    return (
        "<div class='pipeline-summary'>"
        "<div><span>Total users</span><strong>"
        f"{_format_count(total_users)}</strong></div>"
        "<div><span>Generated</span><strong>"
        f"{html.escape(str(generated_at))}</strong></div>"
        "<div><span>Schema</span><strong>"
        f"{html.escape(str(schema_version))}</strong></div>"
        "<div><span>Kanto frame</span><strong>"
        f"{html.escape(', '.join(map(str, kanto_prefectures)) or 'Default')}</strong></div>"
        "<div><span>Metric rows</span><strong>"
        f"{_format_count(metric_count)}</strong></div>"
        "<div><span>Confusion cells</span><strong>"
        f"{_format_count(confusion_count)}</strong></div>"
        "</div>"
    )


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [column for column in DISPLAY_COLUMNS if column in frame.columns]
    remaining = [
        column
        for column in frame.columns
        if column not in preferred and column != "combined_reasoning"
    ]
    table = frame.loc[:, preferred + remaining].copy()
    for column in NUMERIC_COLUMNS.intersection(table.columns):
        table[column] = table[column].round(4)
    return table


load_dotenv(ROOT / ".env")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env before running the QueryChat app.")

client = ChatGoogle(
    model=os.getenv("QUERYCHAT_GEMINI_MODEL", "gemini-2.5-flash"),
    api_key=gemini_api_key,
)

inference_results = _load_results()
metric_summary = _load_metrics()
confusion_summary = _load_confusion()
metadata = _load_json(METADATA_PATH)

qc = QueryChat(
    inference_results,
    "inference_results",
    client=client,
    greeting=GREETING.read_text(encoding="utf-8"),
    data_description=DATA_DESCRIPTION.read_text(encoding="utf-8"),
    tools=_querychat_tools(),
)


def app_ui(request):
    return ui.page_sidebar(
        qc.sidebar(width=420),
        ui.include_css(STYLES),
        ui.div(
            ui.div(
                ui.output_text("dashboard_title"),
                ui.output_text("dashboard_subtitle"),
                class_="dashboard-heading",
            ),
            ui.div(
                ui.input_action_button(
                    "reset_dashboard",
                    "Reset query",
                    class_="btn btn-outline-secondary btn-sm",
                ),
                class_="dashboard-actions",
            ),
            class_="dashboard-topbar",
        ),
        ui.div(
            ui.navset_tab(
                ui.nav_panel(
                    "Overview",
                    ui.accordion(
                        ui.accordion_panel(
                            "KPI Summary",
                            ui.output_ui("metric_tiles"),
                            value="metrics",
                        ),
                        ui.accordion_panel(
                            "Pipeline Summary",
                            ui.output_ui("pipeline_summary"),
                            value="pipeline",
                        ),
                        id="overview_sections",
                        open=["metrics"],
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Location Predictions"),
                            ui.output_ui("location_distribution"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Age Predictions"),
                            ui.output_ui("age_distribution"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Gender Predictions"),
                            ui.output_ui("gender_distribution"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(4, 4, 4),
                    ),
                    value="overview",
                ),
                ui.nav_panel(
                    "Performance",
                    ui.card(
                        ui.card_header("Metrics"),
                        ui.output_ui("performance_table"),
                        class_="dashboard-card",
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Location Matrix"),
                            ui.output_ui("location_confusion"),
                            class_="dashboard-card",
                        ),
                        ui.card(
                            ui.card_header("Age Matrix"),
                            ui.output_ui("age_confusion"),
                            class_="dashboard-card",
                        ),
                        ui.card(
                            ui.card_header("Gender Matrix"),
                            ui.output_ui("gender_confusion"),
                            class_="dashboard-card",
                        ),
                        col_widths=(4, 4, 4),
                    ),
                    value="performance",
                ),
                ui.nav_panel(
                    "Evidence",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Mean Belief"),
                            ui.output_ui("belief_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        ui.card(
                            ui.card_header("Location DST Evidence"),
                            ui.output_ui("dst_evidence_chart"),
                            class_="dashboard-card chart-card",
                        ),
                        col_widths=(5, 7),
                    ),
                    value="evidence",
                ),
                ui.nav_panel(
                    "Users",
                    ui.card(
                        ui.card_header("Filtered Users"),
                        ui.output_data_frame("users_table"),
                        class_="dashboard-card data-card",
                    ),
                    ui.card(
                        ui.card_header("Reasoning"),
                        ui.output_ui("reasoning_cards"),
                        class_="dashboard-card reasoning-preview-card",
                    ),
                    value="users",
                ),
                ui.nav_panel(
                    "SQL",
                    ui.card(
                        ui.card_header("Current Query"),
                        ui.output_ui("sql_output"),
                        class_="dashboard-card sql-card",
                    ),
                    value="sql",
                ),
                id="dashboard_tab",
                selected="overview",
            ),
            class_="dashboard-tabs",
        ),
        title=ui.span("Inference QueryChat"),
        fillable=False,
        class_="inference-dashboard",
    )


def server(input: Inputs, output: Outputs, session: Session) -> None:
    qc_vals = qc.server(enable_bookmarking=False)

    @reactive.effect
    @reactive.event(input.reset_dashboard)
    def _reset_dashboard() -> None:
        qc_vals.sql.set(None)
        qc_vals.title.set(None)

    @reactive.calc
    def filtered_results() -> pd.DataFrame:
        return _clean_frame(_as_pandas(qc_vals.df()))

    @render.text
    def dashboard_title() -> str:
        return qc_vals.title() or "Demographic Inference Overview"

    @render.text
    def dashboard_subtitle() -> str:
        frame = filtered_results()
        return f"Showing {len(frame):,} of {len(inference_results):,} users"

    @render.ui
    def metric_tiles():
        return ui.HTML(_metric_tiles(filtered_results()))

    @render.ui
    def pipeline_summary():
        return ui.HTML(_pipeline_summary(metadata, metric_summary, confusion_summary))

    @render.ui
    def location_distribution():
        return ui.HTML(_prediction_distribution(filtered_results(), "pred_location_group"))

    @render.ui
    def age_distribution():
        return ui.HTML(_prediction_distribution(filtered_results(), "pred_age_group"))

    @render.ui
    def gender_distribution():
        return ui.HTML(_prediction_distribution(filtered_results(), "pred_gender_group"))

    @render.ui
    def performance_table():
        return ui.HTML(_performance_table(filtered_results()))

    @render.ui
    def location_confusion():
        return ui.HTML(_confusion_matrix(filtered_results(), "location"))

    @render.ui
    def age_confusion():
        return ui.HTML(_confusion_matrix(filtered_results(), "age"))

    @render.ui
    def gender_confusion():
        return ui.HTML(_confusion_matrix(filtered_results(), "gender"))

    @render.ui
    def belief_chart():
        return ui.HTML(_belief_chart(filtered_results()))

    @render.ui
    def dst_evidence_chart():
        return ui.HTML(_dst_evidence_chart(filtered_results()))

    @render.ui
    def reasoning_cards():
        return ui.HTML(_reasoning_cards(filtered_results()))

    @render.ui
    def sql_output():
        sql = qc_vals.sql() or "SELECT * FROM inference_results"
        return ui.HTML(
            "<pre class='sql-block'><code>"
            f"{html.escape(sql)}"
            "</code></pre>"
        )

    @render.data_frame
    def users_table():
        return _display_table(filtered_results())


app = App(app_ui, server)
