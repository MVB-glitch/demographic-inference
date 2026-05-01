# -*- coding: utf-8 -*-
"""
Pipeline orchestrator for the Demographic Inference Pipeline.

Processes users in configurable batches with checkpoint/resume support.
Wires together: data loading → LLM extraction → DST fusion → decision.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .dst_engine import (
    apply_discount,
    dempster_fuse_all,
    decide,
    validate_mass,
)
from .llm_extractor import LLMExtractor


# Age category display map
AGE_CATEGORY_MAP = {
    "18_24": "18-24歳",
    "25_34": "25-34歳",
    "35_44": "35-44歳",
    "45_54": "45-54歳",
    "55_64": "55-64歳",
    "65_plus": "65歳以上",
}


def _process_location(ev: dict) -> dict:
    """
    Process location evidence through DST: discount → fuse → decide.

    Args:
        ev: Raw evidence dict from LLM with mass_A/B/C/D and conf_A/B/C/D.

    Returns:
        Dict with location prediction, belief, and intermediate DST data.
    """
    # 1. Discount each source by its confidence
    mA = apply_discount(
        validate_mass(ev.get("mass_A_posts", {"K": 0, "NK": 0, "Omega": 1})),
        ev.get("conf_A", 0),
    )
    mB = apply_discount(
        validate_mass(ev.get("mass_B_friends", {"K": 0, "NK": 0, "Omega": 1})),
        ev.get("conf_B", 0),
    )
    mC = apply_discount(
        validate_mass(ev.get("mass_C_bio", {"K": 0, "NK": 0, "Omega": 1})),
        ev.get("conf_C", 0),
    )
    mD = apply_discount(
        validate_mass(ev.get("mass_D_images", {"K": 0, "NK": 0, "Omega": 1})),
        ev.get("conf_D", 0),
    )

    # 2. Fuse all sources
    m_final, fusion_metrics = dempster_fuse_all([mA, mB, mC, mD])

    # 3. Decision
    prediction_key, belief = decide(m_final, omega_threshold=0.5)
    prediction_map = {"K": "Kanto", "NK": "Non-Kanto", "不確実": "不確実"}
    prediction = prediction_map.get(prediction_key, prediction_key)

    return {
        "Location_Pred": prediction,
        "Location_Belief": belief,
        "mA": mA, "mB": mB, "mC": mC, "mD": mD,
        "m_final": m_final,
        "max_conflict": fusion_metrics.max_pairwise_conflict,
        "conf_A": ev.get("conf_A", 0),
        "conf_B": ev.get("conf_B", 0),
        "conf_C": ev.get("conf_C", 0),
        "conf_D": ev.get("conf_D", 0),
    }


def _process_demographics(demographics_ev: dict) -> dict:
    """
    Process demographics evidence (LLM-only, no DST fusion).

    Args:
        demographics_ev: Raw evidence dict with mass_age and mass_gender.

    Returns:
        Dict with age/gender predictions and beliefs.
    """
    default_age = {"18_24": 0, "25_34": 0, "35_44": 0, "45_54": 0, "55_64": 0, "65_plus": 0, "Omega": 1}
    default_gender = {"M": 0, "F": 0, "Omega": 1}

    mass_age = validate_mass(demographics_ev.get("mass_age", default_age))
    mass_gender = validate_mass(demographics_ev.get("mass_gender", default_gender))

    # Age decision
    predicted_age = "不確実"
    age_belief = round(mass_age.get("Omega", 1.0), 3)

    if mass_age["Omega"] <= 0.5:
        best_age_key = max(
            (k for k in mass_age if k != "Omega"),
            key=lambda k: mass_age[k],
            default=None,
        )
        if best_age_key:
            predicted_age = AGE_CATEGORY_MAP.get(best_age_key, "不確実")
            age_belief = round(mass_age[best_age_key], 3)

    # Gender decision
    predicted_gender = "不確実"
    gender_belief = round(mass_gender.get("Omega", 1.0), 3)

    if mass_gender["Omega"] <= 0.5:
        if mass_gender["M"] > mass_gender["F"]:
            predicted_gender = "男性"
            gender_belief = round(mass_gender["M"], 3)
        elif mass_gender["F"] > mass_gender["M"]:
            predicted_gender = "女性"
            gender_belief = round(mass_gender["F"], 3)

    return {
        "Age_Pred": predicted_age,
        "Age_Belief": age_belief,
        "Gender_Pred": predicted_gender,
        "Gender_Belief": gender_belief,
        "mass_age": mass_age,
        "mass_gender": mass_gender,
    }


def _load_checkpoint(checkpoint_dir: str) -> list[dict]:
    """Load previously checkpointed results."""
    checkpoint_path = os.path.join(checkpoint_dir, "results_checkpoint.json")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"📂 Checkpoint loaded: {len(data)} users already processed.")
            return data
    return []


def _save_checkpoint(results: list[dict], checkpoint_dir: str):
    """Save current results to checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "results_checkpoint.json")

    # Convert non-serializable values for JSON
    serializable = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, dict):
                entry[k] = {str(dk): float(dv) if isinstance(dv, (int, float)) else dv for dk, dv in v.items()}
            elif isinstance(v, float):
                entry[k] = round(v, 6)
            else:
                entry[k] = v
        serializable.append(entry)

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def run_pipeline(
    users: list[dict],
    config: PipelineConfig,
    resume: bool = True,
) -> list[dict]:
    """
    Run the full inference pipeline on a list of users.

    Processes users in batches of config.batch_size, checkpointing
    after each batch so errors don't lose previous progress.

    Args:
        users: List of user data dicts.
        config: Pipeline configuration.
        resume: If True, resume from checkpoint.

    Returns:
        List of result dictionaries.
    """
    extractor = LLMExtractor(config)

    # Load checkpoint if resuming
    results = []
    start_index = 0
    if resume:
        results = _load_checkpoint(config.checkpoint_dir)
        start_index = len(results)
        if start_index > 0:
            print(f"▶️  Resuming from user #{start_index + 1}")

    # Apply user limit
    users_to_process = users[start_index:]
    if config.user_limit and config.user_limit > start_index:
        users_to_process = users[start_index:config.user_limit]
    elif config.user_limit and config.user_limit <= start_index:
        print(f"✅ Already processed {start_index} users (limit: {config.user_limit}). Nothing to do.")
        return results

    total = len(users_to_process)
    print(f"🚀 Processing {total} users in batches of {config.batch_size}...\n")

    for batch_start in range(0, total, config.batch_size):
        batch_end = min(batch_start + config.batch_size, total)
        batch = users_to_process[batch_start:batch_end]
        batch_num = (batch_start // config.batch_size) + 1
        total_batches = (total + config.batch_size - 1) // config.batch_size

        print(f"━━━ Batch {batch_num}/{total_batches} ━━━")

        for i, user in enumerate(batch):
            global_index = start_index + batch_start + i + 1
            print(f"  [{global_index}] User {user['user_id']} (Truth: {user['ground_truth']})")

            try:
                # Location inference (DST-LLM hybrid)
                ev = extractor.extract_location_evidence(user)
                location_result = _process_location(ev)

                # Demographics inference (LLM-only)
                demographics_ev = extractor.extract_demographics_evidence(user)
                demographics_result = _process_demographics(demographics_ev)

                # Combine reasoning
                loc_reasoning = ev.get("reasoning", "")
                demo_reasoning = demographics_ev.get("reasoning", "")
                combined_reasoning = f"地域推論:\n{loc_reasoning}\n\n年齢・性別推論:\n{demo_reasoning}"

                # Build result entry
                result = {
                    "User": user["user_id"],
                    "Truth": user["ground_truth"],
                    "Truth_Age": user.get("ground_truth_age", "Unknown"),
                    "Truth_Gender": user.get("ground_truth_gender", "Unknown"),
                    "Location_Pred": location_result["Location_Pred"],
                    "Location_Belief": location_result["Location_Belief"],
                    "Age_Pred": demographics_result["Age_Pred"],
                    "Age_Belief": demographics_result["Age_Belief"],
                    "Gender_Pred": demographics_result["Gender_Pred"],
                    "Gender_Belief": demographics_result["Gender_Belief"],
                    # Detailed DST data
                    "mA_K": round(location_result["mA"]["K"], 2),
                    "mA_NK": round(location_result["mA"]["NK"], 2),
                    "mA_Conf": location_result["conf_A"],
                    "mB_K": round(location_result["mB"]["K"], 2),
                    "mB_NK": round(location_result["mB"]["NK"], 2),
                    "mB_Conf": location_result["conf_B"],
                    "mC_K": round(location_result["mC"]["K"], 2),
                    "mC_NK": round(location_result["mC"]["NK"], 2),
                    "mC_Conf": location_result["conf_C"],
                    "mD_K": round(location_result["mD"]["K"], 2),
                    "mD_NK": round(location_result["mD"]["NK"], 2),
                    "mD_Conf": location_result["conf_D"],
                    "Max_Conflict": round(location_result["max_conflict"], 3),
                    "DST_K": round(location_result["m_final"]["K"], 3),
                    "DST_NK": round(location_result["m_final"]["NK"], 3),
                    "DST_Omega": round(location_result["m_final"]["Omega"], 3),
                    "mass_age": demographics_result["mass_age"],
                    "mass_gender": demographics_result["mass_gender"],
                    "Combined_Reasoning": combined_reasoning,
                }
                results.append(result)

                print(f"       → Location: {result['Location_Pred']} ({result['Location_Belief']})")
                print(f"       → Age: {result['Age_Pred']} | Gender: {result['Gender_Pred']}")

            except Exception as e:
                print(f"  ❌ Error processing user {user['user_id']}: {e}")
                results.append({
                    "User": user["user_id"],
                    "Truth": user["ground_truth"],
                    "Truth_Age": user.get("ground_truth_age", "Unknown"),
                    "Truth_Gender": user.get("ground_truth_gender", "Unknown"),
                    "Location_Pred": "Error",
                    "Location_Belief": 0,
                    "Age_Pred": "Error",
                    "Age_Belief": 0,
                    "Gender_Pred": "Error",
                    "Gender_Belief": 0,
                    "Combined_Reasoning": f"Error: {e}",
                })

            # Rate limiting between API calls
            time.sleep(config.rate_limit_delay)

        # Checkpoint after each batch
        _save_checkpoint(results, config.checkpoint_dir)
        print(f"  💾 Batch {batch_num} checkpointed ({len(results)} total users)\n")

    print(f"✅ Pipeline complete. {len(results)} users processed.")
    return results
