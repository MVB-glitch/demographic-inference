# Future Work & Research Directions

### 1. Calibration Study — LLM Confidence vs Actual Accuracy

**Goal:** Determine whether the confidence scores output by the LLM actually correlate with prediction accuracy.

**Why it matters:** The DST discounting step multiplies each evidence source's mass by its confidence score. If the LLM's confidence scores are poorly calibrated (e.g., always outputs 0.7 regardless of actual reliability), the discounting step becomes ineffective.

**How to implement:**

1. **Collect predictions**: Run the pipeline on a large dataset (100+ users) with known ground truth.
2. **Record raw confidence values**: For each user and each evidence source (A/B/C/D), record the LLM's `conf_X` value alongside the source's individual prediction and the final prediction.
3. **Bin by confidence**: Group predictions by confidence intervals (e.g., 0.0-0.2, 0.2-0.4, ..., 0.8-1.0).
4. **Compute actual accuracy per bin**: Within each confidence bin, calculate what percentage of predictions were actually correct.
5. **Plot reliability diagram**: An ideal calibration curve is a diagonal line (confidence = accuracy). Deviations indicate over-confidence or under-confidence.
6. **Apply corrections**: If the LLM is systematically over-confident, apply a calibration function (e.g., Platt scaling or isotonic regression) to map raw confidence to calibrated confidence before discounting.

```python
# Example analysis pseudocode
for bin in confidence_bins:
    users_in_bin = [u for u in results if bin.lo <= u.conf < bin.hi]
    actual_accuracy = sum(u.correct for u in users_in_bin) / len(users_in_bin)
    print(f"Confidence {bin}: Predicted ~{bin.mid}, Actual {actual_accuracy}")
```

### 2. Extended Frame of Discernment — Compound Hypotheses

**Goal:** Move beyond binary location classification (Kanto vs Non-Kanto) to richer regional inference.

**Why it matters:** The current binary frame loses information. A user could be from Tokyo vs Osaka vs Nagoya — these have different evidence signatures that get collapsed into "Kanto/Non-Kanto."

**How to implement:**

1. **Define a richer frame**: `Θ = {Tokyo, Kanagawa, Saitama, Chiba, Osaka, Nagoya, Fukuoka, ...}`
2. **Support compound hypotheses**: DST naturally handles subsets like `{Tokyo, Kanagawa}` (= "Southern Kanto"). Modify `dst_engine.py` to handle non-singleton focal elements.
3. **Update the LLM prompt**: Ask the model to assign mass to individual prefectures and regional groups.
4. **Power set handling**: For N hypotheses, the full power set has 2^N elements. For practical use, restrict to a predefined set of "meaningful" subsets (e.g., regions).

```python
# Example extended mass function
mass = {
    frozenset({"Tokyo"}): 0.3,
    frozenset({"Tokyo", "Kanagawa", "Saitama", "Chiba"}): 0.2,  # Kanto region
    frozenset({"Osaka"}): 0.1,
    "Omega": 0.4,  # Full frame
}
```

**Mathematical change**: Dempster's combination rule already supports this — the intersection of two subsets A ∩ B is computed set-theoretically. The current implementation would need to be extended from dict-key matching to proper set intersection.

### 3. Conflict-Aware Decision Logic

**Goal:** Flag high-conflict users for manual review instead of making potentially incorrect predictions.

**Why it matters:** High conflict between evidence sources often indicates either (a) contradictory evidence that needs human judgment, or (b) a user who deliberately obfuscates their identity. Making a prediction in these cases reduces trust in the system.

**How to implement:**

1. **Define a conflict threshold**: Based on empirical analysis, determine what level of pairwise conflict (e.g., K > 0.3) indicates unreliable fusion.
2. **Add a review flag**: In the results, add a `"needs_review": True` field for high-conflict cases.
3. **Classify conflict sources**: Report which pair of evidence sources produced the highest conflict and why.
4. **Implement alternative combination rules**: For high-conflict scenarios, consider using Yager's Rule (which assigns conflicting mass to Omega instead of normalizing) or Murphy's averaging rule (which averages mass functions before combining).

```python
# Example conflict-aware decision
def decide_with_review(mass, conflict_metrics, conflict_threshold=0.3):
    if conflict_metrics.max_pairwise_conflict > conflict_threshold:
        return {
            "prediction": "REVIEW",
            "reason": f"High conflict ({conflict_metrics.max_pairwise_conflict:.2f}) "
                      f"between sources {conflict_metrics.most_conflicting_pair}",
            "needs_review": True,
        }
    return decide(mass)
```

### 4. Structured Logging

**Goal:** Replace print statements with Python's `logging` module for production-grade observability.

**How to implement:**

1. **Configure logging levels**: Use `DEBUG` for detailed DST math, `INFO` for progress, `WARNING` for API issues, `ERROR` for failures.
2. **Add file handler**: Log to `pipeline.log` in addition to console.
3. **Structured format**: Include timestamps, user IDs, and batch numbers.

```python
import logging

logger = logging.getLogger("demographic_inference")

# In pipeline.py
logger.info(f"Processing user {user_id}", extra={"user_id": user_id, "batch": batch_num})
logger.debug(f"DST fusion result: K={m_final['K']:.3f}, NK={m_final['NK']:.3f}")
logger.warning(f"High conflict detected: {max_conflict:.3f}")
```

### 5. Multi-Source Demographics DST

**Goal:** Apply the same DST multi-source fusion approach to age and gender inference.

**How to implement:** Split the demographics LLM call into 3 independent evidence sources (text-based, image-based, bio/handle-based), extract mass functions from each independently, and fuse with Dempster's Rule — exactly mirroring the location pipeline.

### 6. Ensemble LLM Fusion

**Goal:** Use multiple LLM calls (or different models) as independent evidence sources and fuse via DST.

**How to implement:** Run the same prompt through 2-3 different models (e.g., Gemini Flash, Gemini Pro, GPT-4o), treat each model's output as an independent evidence source, and combine using Dempster's Rule. This provides natural robustness against individual model biases.
