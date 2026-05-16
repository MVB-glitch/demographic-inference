from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demographic_inference.querychat_export import build_querychat_payload  # noqa: E402


AGE_KEYS = ["18_24", "25_34", "35_44", "45_54", "55_64", "65_plus"]
AGE_LABELS = {
    "18_24": "18-24\u6b73",
    "25_34": "25-34\u6b73",
    "35_44": "35-44\u6b73",
    "45_54": "45-54\u6b73",
    "55_64": "55-64\u6b73",
    "65_plus": "65\u6b73\u4ee5\u4e0a",
}
TRUTH_AGE_LABELS = {
    "18_24": "18-24",
    "25_34": "25-34",
    "35_44": "35-44",
    "45_54": "45-54",
    "55_64": "55-64",
    "65_plus": "65+",
}
KANTO_PREFECTURES = ["Tokyo", "Saitama", "Chiba", "Kanagawa", "Ibaraki", "Tochigi", "Gunma"]
NON_KANTO_PREFECTURES = [
    "Osaka",
    "Kyoto",
    "Hyogo",
    "Aichi",
    "Fukuoka",
    "Hokkaido",
    "Miyagi",
    "Hiroshima",
    "Okinawa",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic demographic inference results.json file.",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "results.json",
        help="Output JSON path. Defaults to results.json in the repository root.",
    )
    parser.add_argument("--users", type=int, default=100, help="Number of fake users.")
    parser.add_argument("--seed", type=int, default=937, help="Random seed.")
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _softmax_like(keys: list[str], preferred: str | None, accuracy: float) -> dict[str, float]:
    values = {key: 0.0 for key in keys}
    if preferred is None:
        omega = random.uniform(0.55, 0.85)
        remaining = 1.0 - omega
        for key in keys:
            values[key] = remaining / len(keys)
        values["Omega"] = omega
        return {key: round(value, 3) for key, value in values.items()}

    omega = random.uniform(0.08, 0.32)
    preferred_mass = random.uniform(max(0.42, accuracy - 0.16), min(0.82, accuracy + 0.12))
    preferred_mass = min(preferred_mass, 1.0 - omega)
    remaining = 1.0 - omega - preferred_mass
    other_keys = [key for key in keys if key != preferred]
    weights = [random.uniform(0.4, 1.6) for _ in other_keys]
    total_weight = sum(weights)
    values[preferred] = preferred_mass
    for key, weight in zip(other_keys, weights):
        values[key] = remaining * (weight / total_weight)
    values["Omega"] = omega
    return {key: round(value, 3) for key, value in values.items()}


def _location_source_mass(truth_is_kanto: bool, signal_quality: float) -> tuple[dict[str, float], float]:
    confidence = round(random.uniform(0.35, 0.95) * signal_quality, 3)
    supports_truth = random.random() < signal_quality
    preferred = "K" if truth_is_kanto == supports_truth else "NK"
    if not truth_is_kanto and supports_truth:
        preferred = "NK"
    if truth_is_kanto and not supports_truth:
        preferred = "NK"
    if not truth_is_kanto and not supports_truth:
        preferred = "K"

    specific = random.uniform(0.32, 0.78) * confidence
    opposite = random.uniform(0.0, 0.22) * confidence
    omega = max(0.0, 1.0 - specific - opposite)
    if preferred == "K":
        mass = {"K": specific, "NK": opposite, "Omega": omega}
    else:
        mass = {"K": opposite, "NK": specific, "Omega": omega}
    return {key: round(value, 3) for key, value in mass.items()}, confidence


def _decide_location(masses: list[dict[str, float]], truth_is_kanto: bool) -> tuple[str, float, float, float, float]:
    k_score = sum(mass["K"] for mass in masses) / len(masses)
    nk_score = sum(mass["NK"] for mass in masses) / len(masses)
    conflict = min(0.95, abs(masses[0]["K"] - masses[1]["NK"]) * random.uniform(0.3, 0.9))
    omega = max(0.05, min(0.62, sum(mass["Omega"] for mass in masses) / len(masses)))

    if omega > 0.68 or random.random() < 0.08:
        return "\u4e0d\u78ba\u5b9f", omega, k_score, nk_score, omega

    predicted_kanto = k_score >= nk_score
    if random.random() < 0.14:
        predicted_kanto = not predicted_kanto

    pred = "Kanto" if predicted_kanto else "Non-Kanto"
    belief = max(k_score, nk_score)
    if predicted_kanto == truth_is_kanto:
        belief = max(belief, random.uniform(0.52, 0.88))
    else:
        belief = max(belief, random.uniform(0.42, 0.7))

    total = max(k_score + nk_score + omega, 1e-9)
    return pred, round(belief, 3), round(k_score / total, 3), round(nk_score / total, 3), round(omega / total, 3)


def _nearby_age(age_key: str) -> str:
    index = AGE_KEYS.index(age_key)
    candidates = [age_key]
    if index > 0:
        candidates.append(AGE_KEYS[index - 1])
    if index < len(AGE_KEYS) - 1:
        candidates.append(AGE_KEYS[index + 1])
    return random.choice(candidates)


def _reasoning(user_id: str, location: str, age: str, gender: str, conflict: float) -> str:
    return (
        f"Location inference for {user_id} combines posts, profile/network evidence, "
        f"and image cues. The fused DST result favors {location} with conflict {conflict:.3f}. "
        f"Age and gender are LLM-only estimates, with the strongest demographic cues pointing "
        f"to {age} and {gender}."
    )


def _generate_raw_results(user_count: int, seed: int) -> list[dict[str, Any]]:
    random.seed(seed)
    rows: list[dict[str, Any]] = []

    for index in range(1, user_count + 1):
        user_id = f"synthetic_{index:03d}"
        truth_is_kanto = random.random() < 0.58
        truth_location = random.choice(KANTO_PREFECTURES if truth_is_kanto else NON_KANTO_PREFECTURES)
        truth_age_key = random.choices(
            AGE_KEYS,
            weights=[14, 24, 24, 18, 13, 7],
            k=1,
        )[0]
        truth_gender = random.choice(["M", "F"])

        source_qualities = [
            random.uniform(0.62, 0.95),
            random.uniform(0.38, 0.82),
            random.uniform(0.55, 0.9),
            random.uniform(0.18, 0.72),
        ]
        source_data = [_location_source_mass(truth_is_kanto, quality) for quality in source_qualities]
        masses = [item[0] for item in source_data]
        confidences = [item[1] for item in source_data]
        location_pred, location_belief, dst_k, dst_nk, dst_omega = _decide_location(masses, truth_is_kanto)
        max_conflict = round(random.uniform(0.02, 0.48), 3)
        if location_pred != "Kanto" and location_pred != "Non-Kanto":
            max_conflict = round(random.uniform(0.2, 0.82), 3)

        age_uncertain = random.random() < 0.09
        pred_age_key = None if age_uncertain else _nearby_age(truth_age_key)
        if pred_age_key == truth_age_key and random.random() < 0.17:
            pred_age_key = random.choice([key for key in AGE_KEYS if key != truth_age_key])
        mass_age = _softmax_like(AGE_KEYS, pred_age_key, random.uniform(0.55, 0.82))
        age_pred = "\u4e0d\u78ba\u5b9f" if pred_age_key is None else AGE_LABELS[pred_age_key]
        age_belief = mass_age["Omega"] if pred_age_key is None else mass_age[pred_age_key]

        gender_uncertain = random.random() < 0.07
        if gender_uncertain:
            pred_gender_key = None
        elif random.random() < 0.84:
            pred_gender_key = truth_gender
        else:
            pred_gender_key = "F" if truth_gender == "M" else "M"
        mass_gender = _softmax_like(["M", "F"], pred_gender_key, random.uniform(0.58, 0.86))
        gender_pred = (
            "\u4e0d\u78ba\u5b9f"
            if pred_gender_key is None
            else "\u7537\u6027"
            if pred_gender_key == "M"
            else "\u5973\u6027"
        )
        gender_belief = mass_gender["Omega"] if pred_gender_key is None else mass_gender[pred_gender_key]

        rows.append(
            {
                "User": user_id,
                "Truth": truth_location,
                "Truth_Age": TRUTH_AGE_LABELS[truth_age_key],
                "Truth_Gender": truth_gender,
                "Location_Pred": location_pred,
                "Location_Belief": round(location_belief, 3),
                "Age_Pred": age_pred,
                "Age_Belief": round(age_belief, 3),
                "Gender_Pred": gender_pred,
                "Gender_Belief": round(gender_belief, 3),
                "mA_K": masses[0]["K"],
                "mA_NK": masses[0]["NK"],
                "mA_Conf": confidences[0],
                "mB_K": masses[1]["K"],
                "mB_NK": masses[1]["NK"],
                "mB_Conf": confidences[1],
                "mC_K": masses[2]["K"],
                "mC_NK": masses[2]["NK"],
                "mC_Conf": confidences[2],
                "mD_K": masses[3]["K"],
                "mD_NK": masses[3]["NK"],
                "mD_Conf": confidences[3],
                "Max_Conflict": max_conflict,
                "DST_K": dst_k,
                "DST_NK": dst_nk,
                "DST_Omega": dst_omega,
                "mass_age": mass_age,
                "mass_gender": mass_gender,
                "Combined_Reasoning": _reasoning(
                    user_id,
                    location_pred,
                    age_pred,
                    gender_pred,
                    max_conflict,
                ),
            }
        )

    return rows


def main() -> None:
    args = _parse_args()
    output = _resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_results = _generate_raw_results(args.users, args.seed)
    payload = build_querychat_payload(
        raw_results,
        kanto_prefectures=KANTO_PREFECTURES,
        source_file=output,
    )

    with output.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)

    print(f"Wrote {args.users:,} synthetic users to: {output}")
    print(
        "Accuracy: "
        f"location={payload['metrics']['location']['accuracy']}, "
        f"age={payload['metrics']['age']['accuracy']}, "
        f"gender={payload['metrics']['gender']['accuracy']}"
    )


if __name__ == "__main__":
    main()
