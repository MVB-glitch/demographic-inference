# -*- coding: utf-8 -*-
"""
Metrics computation and result reporting.

Computes accuracy for location, age, and gender predictions.
Exports results to Excel, CSV, or JSON.
"""

import json
from typing import Optional

import pandas as pd

from .querychat_export import build_querychat_payload


def compute_location_accuracy(
    results: list[dict],
    kanto_prefectures: Optional[list[str]] = None,
) -> dict:
    """
    Compute accuracy for location predictions.

    Only evaluates definitive predictions (excludes "不確実" and "Error").

    Args:
        results: List of result dicts from the pipeline.
        kanto_prefectures: List of Kanto prefecture names for ground truth matching.

    Returns:
        Dict with accuracy metrics.
    """
    if kanto_prefectures is None:
        kanto_prefectures = ["Tokyo", "Saitama", "Chiba", "Kanagawa", "Ibaraki", "Tochigi", "Gunma"]

    total = len(results)
    evaluable = 0
    correct = 0
    uncertain_count = 0

    for r in results:
        pred = r.get("Location_Pred", "")
        if pred in ("不確実", "Error"):
            uncertain_count += 1
            continue

        evaluable += 1
        is_truth_kanto = r["Truth"] in kanto_prefectures
        is_pred_kanto = pred == "Kanto"

        if (is_pred_kanto and is_truth_kanto) or (not is_pred_kanto and not is_truth_kanto):
            correct += 1

    return {
        "total": total,
        "evaluable": evaluable,
        "correct": correct,
        "uncertain": uncertain_count,
        "accuracy": correct / evaluable if evaluable > 0 else 0.0,
    }


def compute_demographics_accuracy(results: list[dict]) -> dict:
    """
    Compute accuracy for age and gender predictions.

    Only evaluates users with known ground truth (not "Unknown") and
    definitive predictions (not "不確実" or "Error").

    Returns:
        Dict with age and gender accuracy metrics.
    """
    # Age accuracy
    age_total = 0
    age_correct = 0
    age_uncertain = 0

    # Gender accuracy
    gender_total = 0
    gender_correct = 0
    gender_uncertain = 0

    age_truth_map = {
        "18-24": "18-24歳", "25-34": "25-34歳", "35-44": "35-44歳",
        "45-54": "45-54歳", "55-64": "55-64歳", "65+": "65歳以上",
        "18-24歳": "18-24歳", "25-34歳": "25-34歳", "35-44歳": "35-44歳",
        "45-54歳": "45-54歳", "55-64歳": "55-64歳", "65歳以上": "65歳以上",
    }

    gender_truth_map = {
        "M": "男性", "F": "女性", "Male": "男性", "Female": "女性",
        "男性": "男性", "女性": "女性",
    }

    for r in results:
        # Age
        truth_age = r.get("Truth_Age", "Unknown")
        if truth_age != "Unknown":
            pred_age = r.get("Age_Pred", "")
            if pred_age in ("不確実", "Error"):
                age_uncertain += 1
            else:
                age_total += 1
                normalized_truth = age_truth_map.get(truth_age, truth_age)
                if pred_age == normalized_truth:
                    age_correct += 1

        # Gender
        truth_gender = r.get("Truth_Gender", "Unknown")
        if truth_gender != "Unknown":
            pred_gender = r.get("Gender_Pred", "")
            if pred_gender in ("不確実", "Error"):
                gender_uncertain += 1
            else:
                gender_total += 1
                normalized_truth = gender_truth_map.get(truth_gender, truth_gender)
                if pred_gender == normalized_truth:
                    gender_correct += 1

    return {
        "age": {
            "evaluable": age_total,
            "correct": age_correct,
            "uncertain": age_uncertain,
            "accuracy": age_correct / age_total if age_total > 0 else 0.0,
        },
        "gender": {
            "evaluable": gender_total,
            "correct": gender_correct,
            "uncertain": gender_uncertain,
            "accuracy": gender_correct / gender_total if gender_total > 0 else 0.0,
        },
    }


def print_metrics(results: list[dict], kanto_prefectures: Optional[list[str]] = None):
    """Print accuracy metrics to console."""
    loc = compute_location_accuracy(results, kanto_prefectures)
    demo = compute_demographics_accuracy(results)

    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)

    print(f"\n🎯 Location Accuracy: {loc['correct']}/{loc['evaluable']} "
          f"({loc['accuracy']*100:.1f}%) "
          f"[{loc['uncertain']} uncertain]")

    if demo["age"]["evaluable"] > 0:
        print(f"🎂 Age Accuracy: {demo['age']['correct']}/{demo['age']['evaluable']} "
              f"({demo['age']['accuracy']*100:.1f}%) "
              f"[{demo['age']['uncertain']} uncertain]")

    if demo["gender"]["evaluable"] > 0:
        print(f"👤 Gender Accuracy: {demo['gender']['correct']}/{demo['gender']['evaluable']} "
              f"({demo['gender']['accuracy']*100:.1f}%) "
              f"[{demo['gender']['uncertain']} uncertain]")


def generate_summary_dataframe(results: list[dict]) -> pd.DataFrame:
    """Create a summary DataFrame with key columns for display."""
    summary_columns = [
        "User", "Truth", "Truth_Age", "Truth_Gender",
        "Location_Pred", "Location_Belief",
        "Age_Pred", "Age_Belief",
        "Gender_Pred", "Gender_Belief",
        "Combined_Reasoning",
    ]
    df = pd.DataFrame(results)
    available = [c for c in summary_columns if c in df.columns]
    return df[available]


def export_results(
    results: list[dict],
    output_path: str,
    output_format: str = "xlsx",
    kanto_prefectures: Optional[list[str]] = None,
):
    """
    Export results to file.

    Args:
        results: List of result dicts.
        output_path: Output file path.
        output_format: One of 'xlsx', 'csv', 'json'.
    """
    df = pd.DataFrame(results)

    if output_format == "xlsx":
        df.to_excel(output_path, index=False)
    elif output_format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif output_format == "json":
        payload = build_querychat_payload(
            results,
            kanto_prefectures=kanto_prefectures,
            source_file=output_path,
        )
        with open(output_path, "w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print(f"💾 Results saved to: {output_path}")
