# -*- coding: utf-8 -*-
"""Structured export helpers for the QueryChat inference explorer."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_KANTO_PREFECTURES = [
    "Tokyo",
    "Saitama",
    "Chiba",
    "Kanagawa",
    "Ibaraki",
    "Tochigi",
    "Gunma",
]

UNCERTAIN_VALUES = {
    "",
    "unknown",
    "uncertain",
    "error",
    "nan",
    "none",
    "n/a",
    "\u4e0d\u78ba\u5b9f",
}

AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
AGE_MIDPOINTS = {
    "18-24": 21.0,
    "25-34": 29.5,
    "35-44": 39.5,
    "45-54": 49.5,
    "55-64": 59.5,
    "65+": 70.0,
}
AGE_ALIASES = {
    "18_24": "18-24",
    "18-24": "18-24",
    "18-24\u6b73": "18-24",
    "25_34": "25-34",
    "25-34": "25-34",
    "25-34\u6b73": "25-34",
    "35_44": "35-44",
    "35-44": "35-44",
    "35-44\u6b73": "35-44",
    "45_54": "45-54",
    "45-54": "45-54",
    "45-54\u6b73": "45-54",
    "55_64": "55-64",
    "55-64": "55-64",
    "55-64\u6b73": "55-64",
    "65_plus": "65+",
    "65+": "65+",
    "65\u6b73\u4ee5\u4e0a": "65+",
}

GENDER_ALIASES = {
    "m": "Male",
    "male": "Male",
    "\u7537\u6027": "Male",
    "f": "Female",
    "female": "Female",
    "\u5973\u6027": "Female",
}

DIMENSION_CONFIG = {
    "location": {
        "truth": "truth_location_group",
        "prediction": "pred_location_group",
        "labels": ["Kanto", "Non-Kanto"],
    },
    "age": {
        "truth": "truth_age_group",
        "prediction": "pred_age_group",
        "labels": AGE_ORDER,
    },
    "gender": {
        "truth": "truth_gender_group",
        "prediction": "pred_gender_group",
        "labels": ["Male", "Female"],
    },
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _is_uncertain(value: Any) -> bool:
    return _clean_text(value).lower() in UNCERTAIN_VALUES


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _round_optional(value: Any, digits: int = 4) -> Optional[float]:
    numeric = _safe_float(value)
    return None if numeric is None else round(numeric, digits)


def _sum_to_omega(k_value: Any, nk_value: Any) -> Optional[float]:
    k = _safe_float(k_value)
    nk = _safe_float(nk_value)
    if k is None or nk is None:
        return None
    return round(max(0.0, 1.0 - k - nk), 4)


def normalize_location_truth(
    value: Any,
    kanto_prefectures: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Normalize a prefecture truth label to Kanto/Non-Kanto."""
    if _is_uncertain(value):
        return None
    prefectures = set(kanto_prefectures or DEFAULT_KANTO_PREFECTURES)
    return "Kanto" if _clean_text(value) in prefectures else "Non-Kanto"


def normalize_location_prediction(value: Any) -> Optional[str]:
    text = _clean_text(value)
    if text in {"Kanto", "Non-Kanto"}:
        return text
    return None


def normalize_age(value: Any) -> Optional[str]:
    if _is_uncertain(value):
        return None
    text = _clean_text(value)
    if text in AGE_ALIASES:
        return AGE_ALIASES[text]
    normalized = text.lower().replace(" ", "").replace("_", "-")
    if normalized in AGE_ALIASES:
        return AGE_ALIASES[normalized]
    for label in AGE_ORDER:
        if normalized.startswith(label):
            return label
    if normalized.startswith("65") and ("+" in normalized or "\u4ee5\u4e0a" in normalized):
        return "65+"
    return None


def normalize_gender(value: Any) -> Optional[str]:
    if _is_uncertain(value):
        return None
    return GENDER_ALIASES.get(_clean_text(value).lower())


def _mass_value(mass: Any, key: str) -> Optional[float]:
    if not isinstance(mass, dict):
        return None
    return _round_optional(mass.get(key))


def _prediction_state(truth: Optional[str], prediction: Optional[str]) -> Optional[bool]:
    if truth is None or prediction is None:
        return None
    return truth == prediction


def normalize_result_record(
    result: dict[str, Any],
    kanto_prefectures: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Flatten one pipeline result into a QueryChat-friendly row."""
    truth_location_group = normalize_location_truth(result.get("Truth"), kanto_prefectures)
    pred_location_group = normalize_location_prediction(result.get("Location_Pred"))
    truth_age_group = normalize_age(result.get("Truth_Age"))
    pred_age_group = normalize_age(result.get("Age_Pred"))
    truth_gender_group = normalize_gender(result.get("Truth_Gender"))
    pred_gender_group = normalize_gender(result.get("Gender_Pred"))

    age_abs_error = None
    if truth_age_group and pred_age_group:
        age_abs_error = abs(AGE_MIDPOINTS[truth_age_group] - AGE_MIDPOINTS[pred_age_group])

    mass_age = result.get("mass_age", {})
    mass_gender = result.get("mass_gender", {})

    record = {
        "user_id": _clean_text(result.get("User")),
        "truth_location_raw": _clean_text(result.get("Truth")),
        "truth_location_group": truth_location_group,
        "pred_location_raw": _clean_text(result.get("Location_Pred")),
        "pred_location_group": pred_location_group,
        "location_correct": _prediction_state(truth_location_group, pred_location_group),
        "location_belief": _round_optional(result.get("Location_Belief")),
        "truth_age_raw": _clean_text(result.get("Truth_Age")),
        "truth_age_group": truth_age_group,
        "pred_age_raw": _clean_text(result.get("Age_Pred")),
        "pred_age_group": pred_age_group,
        "age_correct": _prediction_state(truth_age_group, pred_age_group),
        "age_abs_error_years": _round_optional(age_abs_error, 2),
        "age_belief": _round_optional(result.get("Age_Belief")),
        "truth_gender_raw": _clean_text(result.get("Truth_Gender")),
        "truth_gender_group": truth_gender_group,
        "pred_gender_raw": _clean_text(result.get("Gender_Pred")),
        "pred_gender_group": pred_gender_group,
        "gender_correct": _prediction_state(truth_gender_group, pred_gender_group),
        "gender_belief": _round_optional(result.get("Gender_Belief")),
        "dst_k": _round_optional(result.get("DST_K")),
        "dst_non_kanto": _round_optional(result.get("DST_NK")),
        "dst_omega": _round_optional(result.get("DST_Omega")),
        "max_conflict": _round_optional(result.get("Max_Conflict")),
        "posts_kanto_mass": _round_optional(result.get("mA_K")),
        "posts_non_kanto_mass": _round_optional(result.get("mA_NK")),
        "posts_omega_mass": _sum_to_omega(result.get("mA_K"), result.get("mA_NK")),
        "posts_confidence": _round_optional(result.get("mA_Conf")),
        "network_kanto_mass": _round_optional(result.get("mB_K")),
        "network_non_kanto_mass": _round_optional(result.get("mB_NK")),
        "network_omega_mass": _sum_to_omega(result.get("mB_K"), result.get("mB_NK")),
        "network_confidence": _round_optional(result.get("mB_Conf")),
        "bio_kanto_mass": _round_optional(result.get("mC_K")),
        "bio_non_kanto_mass": _round_optional(result.get("mC_NK")),
        "bio_omega_mass": _sum_to_omega(result.get("mC_K"), result.get("mC_NK")),
        "bio_confidence": _round_optional(result.get("mC_Conf")),
        "images_kanto_mass": _round_optional(result.get("mD_K")),
        "images_non_kanto_mass": _round_optional(result.get("mD_NK")),
        "images_omega_mass": _sum_to_omega(result.get("mD_K"), result.get("mD_NK")),
        "images_confidence": _round_optional(result.get("mD_Conf")),
        "age_mass_18_24": _mass_value(mass_age, "18_24"),
        "age_mass_25_34": _mass_value(mass_age, "25_34"),
        "age_mass_35_44": _mass_value(mass_age, "35_44"),
        "age_mass_45_54": _mass_value(mass_age, "45_54"),
        "age_mass_55_64": _mass_value(mass_age, "55_64"),
        "age_mass_65_plus": _mass_value(mass_age, "65_plus"),
        "age_mass_omega": _mass_value(mass_age, "Omega"),
        "gender_mass_male": _mass_value(mass_gender, "M"),
        "gender_mass_female": _mass_value(mass_gender, "F"),
        "gender_mass_omega": _mass_value(mass_gender, "Omega"),
        "combined_reasoning": _clean_text(result.get("Combined_Reasoning")),
    }
    record["correct_dimensions"] = sum(
        1
        for value in (
            record["location_correct"],
            record["age_correct"],
            record["gender_correct"],
        )
        if value is True
    )
    record["evaluated_dimensions"] = sum(
        1
        for value in (
            record["location_correct"],
            record["age_correct"],
            record["gender_correct"],
        )
        if value is not None
    )
    return record


def _macro_prf(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Optional[float]]:
    precision_values = []
    recall_values = []
    f1_values = []
    support_by_label = Counter(row["truth"] for row in rows)

    for label in labels:
        tp = sum(1 for row in rows if row["truth"] == label and row["prediction"] == label)
        fp = sum(1 for row in rows if row["truth"] != label and row["prediction"] == label)
        fn = sum(1 for row in rows if row["truth"] == label and row["prediction"] != label)

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall)
            else None
        )

        if support_by_label[label] > 0:
            if precision is not None:
                precision_values.append(precision)
            if recall is not None:
                recall_values.append(recall)
            if f1 is not None:
                f1_values.append(f1)

    return {
        "precision": round(sum(precision_values) / len(precision_values), 4)
        if precision_values
        else None,
        "recall": round(sum(recall_values) / len(recall_values), 4)
        if recall_values
        else None,
        "f1": round(sum(f1_values) / len(f1_values), 4) if f1_values else None,
    }


def _classification_rows(records: list[dict[str, Any]], dimension: str) -> list[dict[str, str]]:
    config = DIMENSION_CONFIG[dimension]
    rows = []
    for record in records:
        truth = record.get(config["truth"])
        prediction = record.get(config["prediction"])
        if truth is None or prediction is None:
            continue
        rows.append({"truth": truth, "prediction": prediction})
    return rows


def _classification_metrics(records: list[dict[str, Any]], dimension: str) -> dict[str, Any]:
    config = DIMENSION_CONFIG[dimension]
    labels = config["labels"]
    rows = _classification_rows(records, dimension)
    truth_count = sum(1 for record in records if record.get(config["truth"]) is not None)
    correct = sum(1 for row in rows if row["truth"] == row["prediction"])
    prf = _macro_prf(rows, labels)

    metrics = {
        "truth_count": truth_count,
        "evaluable": len(rows),
        "correct": correct,
        "uncertain_or_missing_prediction": max(0, truth_count - len(rows)),
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "coverage": round(len(rows) / truth_count, 4) if truth_count else None,
    }

    if dimension == "age":
        errors = [
            record["age_abs_error_years"]
            for record in records
            if record.get("age_abs_error_years") is not None
        ]
        metrics["mae_years"] = round(sum(errors) / len(errors), 2) if errors else None

    return metrics


def _confusion_matrix(records: list[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    config = DIMENSION_CONFIG[dimension]
    labels = config["labels"]
    rows = _classification_rows(records, dimension)
    counts = Counter((row["truth"], row["prediction"]) for row in rows)
    return [
        {
            "dimension": dimension,
            "truth": truth,
            "prediction": prediction,
            "count": counts[(truth, prediction)],
        }
        for truth in labels
        for prediction in labels
    ]


def _distribution(records: list[dict[str, Any]], column: str) -> list[dict[str, Any]]:
    counts = Counter(record.get(column) or "Uncertain/Missing" for record in records)
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _metric_rows(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dimension, values in metrics.items():
        if dimension == "overall":
            continue
        for metric, value in values.items():
            rows.append({"dimension": dimension, "metric": metric, "value": value})
    return rows


def build_querychat_payload(
    results: list[dict[str, Any]],
    kanto_prefectures: Optional[Iterable[str]] = None,
    source_file: str | Path | None = None,
) -> dict[str, Any]:
    """Build the structured JSON consumed by the QueryChat explorer."""
    records = [
        normalize_result_record(result, kanto_prefectures=kanto_prefectures)
        for result in results
    ]
    metrics = {
        "overall": {
            "total_users": len(records),
            "with_location_truth": sum(
                1 for record in records if record["truth_location_group"] is not None
            ),
            "with_age_truth": sum(1 for record in records if record["truth_age_group"] is not None),
            "with_gender_truth": sum(
                1 for record in records if record["truth_gender_group"] is not None
            ),
            "mean_location_belief": _mean(record["location_belief"] for record in records),
            "mean_age_belief": _mean(record["age_belief"] for record in records),
            "mean_gender_belief": _mean(record["gender_belief"] for record in records),
            "mean_max_conflict": _mean(record["max_conflict"] for record in records),
        },
        "location": _classification_metrics(records, "location"),
        "age": _classification_metrics(records, "age"),
        "gender": _classification_metrics(records, "gender"),
    }
    confusion = {
        "location": _confusion_matrix(records, "location"),
        "age": _confusion_matrix(records, "age"),
        "gender": _confusion_matrix(records, "gender"),
    }
    flat_confusion = [
        row
        for dimension_rows in confusion.values()
        for row in dimension_rows
    ]

    payload = {
        "metadata": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source_file) if source_file else None,
            "total_users": len(records),
            "kanto_prefectures": list(kanto_prefectures or DEFAULT_KANTO_PREFECTURES),
        },
        "results": records,
        "metrics": metrics,
        "metric_rows": _metric_rows(metrics),
        "confusion_matrices": confusion,
        "confusion_rows": flat_confusion,
        "distributions": {
            "location_predictions": _distribution(records, "pred_location_group"),
            "age_predictions": _distribution(records, "pred_age_group"),
            "gender_predictions": _distribution(records, "pred_gender_group"),
            "location_truth": _distribution(records, "truth_location_group"),
            "age_truth": _distribution(records, "truth_age_group"),
            "gender_truth": _distribution(records, "truth_gender_group"),
        },
    }
    return payload


def _mean(values: Iterable[Any]) -> Optional[float]:
    numeric = [value for value in (_safe_float(value) for value in values) if value is not None]
    return round(sum(numeric) / len(numeric), 4) if numeric else None

