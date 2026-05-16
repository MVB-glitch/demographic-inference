# -*- coding: utf-8 -*-
"""Tests for QueryChat structured export helpers."""

from demographic_inference.querychat_export import (
    build_querychat_payload,
    normalize_age,
    normalize_gender,
    normalize_location_prediction,
    normalize_location_truth,
)


def _sample_results():
    return [
        {
            "User": "u001",
            "Truth": "Tokyo",
            "Truth_Age": "25-34",
            "Truth_Gender": "M",
            "Location_Pred": "Kanto",
            "Location_Belief": 0.82,
            "Age_Pred": "25-34\u6b73",
            "Age_Belief": 0.7,
            "Gender_Pred": "\u7537\u6027",
            "Gender_Belief": 0.8,
            "mA_K": 0.5,
            "mA_NK": 0.1,
            "mA_Conf": 0.9,
            "mB_K": 0.2,
            "mB_NK": 0.1,
            "mB_Conf": 0.6,
            "mC_K": 0.4,
            "mC_NK": 0.0,
            "mC_Conf": 0.8,
            "mD_K": 0.0,
            "mD_NK": 0.0,
            "mD_Conf": 0.0,
            "Max_Conflict": 0.12,
            "DST_K": 0.82,
            "DST_NK": 0.08,
            "DST_Omega": 0.1,
            "mass_age": {"25_34": 0.7, "Omega": 0.3},
            "mass_gender": {"M": 0.8, "F": 0.05, "Omega": 0.15},
            "Combined_Reasoning": "Reasoning text",
        },
        {
            "User": "u002",
            "Truth": "Osaka",
            "Truth_Age": "35-44",
            "Truth_Gender": "F",
            "Location_Pred": "Kanto",
            "Location_Belief": 0.55,
            "Age_Pred": "45-54\u6b73",
            "Age_Belief": 0.62,
            "Gender_Pred": "\u4e0d\u78ba\u5b9f",
            "Gender_Belief": 0.7,
            "Max_Conflict": 0.4,
            "DST_K": 0.55,
            "DST_NK": 0.2,
            "DST_Omega": 0.25,
            "mass_age": {"45_54": 0.62, "Omega": 0.2},
            "mass_gender": {"M": 0.2, "F": 0.2, "Omega": 0.6},
            "Combined_Reasoning": "Uncertain gender",
        },
    ]


def test_normalizers():
    assert normalize_location_truth("Tokyo") == "Kanto"
    assert normalize_location_truth("Osaka") == "Non-Kanto"
    assert normalize_location_prediction("Non-Kanto") == "Non-Kanto"
    assert normalize_age("65\u6b73\u4ee5\u4e0a") == "65+"
    assert normalize_gender("\u5973\u6027") == "Female"


def test_build_querychat_payload_metrics_and_rows():
    payload = build_querychat_payload(_sample_results(), source_file="results.json")

    assert payload["metadata"]["total_users"] == 2
    assert len(payload["results"]) == 2
    assert payload["results"][0]["location_correct"] is True
    assert payload["results"][1]["location_correct"] is False
    assert payload["results"][1]["gender_correct"] is None
    assert payload["results"][1]["age_abs_error_years"] == 10.0

    assert payload["metrics"]["location"]["accuracy"] == 0.5
    assert payload["metrics"]["age"]["accuracy"] == 0.5
    assert payload["metrics"]["age"]["mae_years"] == 5.0
    assert payload["metrics"]["gender"]["coverage"] == 0.5

    assert any(row["dimension"] == "location" for row in payload["confusion_rows"])
    assert payload["distributions"]["location_predictions"][0]["label"] == "Kanto"
