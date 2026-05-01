# -*- coding: utf-8 -*-
"""
Generalized Dempster-Shafer Theory (DST) Engine.

Provides mathematical operations for evidence combination using DST:
- Mass function normalization (any frame of discernment)
- Evidence discounting (Shafer's discount operation)
- Dempster's Rule of Combination (generalized for arbitrary singleton frames + Omega)
- Sequential multi-source fusion
- Belief and Plausibility interval computation
- Decision function with uncertainty threshold

Mathematical Reference:
    Frame of discernment Θ = {H1, H2, ..., Hn}
    Power set 2^Θ includes singletons {Hi} and the full frame Θ (= Omega).
    We use a simplified representation where only singletons and Omega carry mass.
    This is valid when evidence only supports individual hypotheses or total ignorance.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FusionMetrics:
    """Metrics collected during sequential DST fusion."""
    max_pairwise_conflict: float = 0.0
    total_conflict_product: float = 0.0  # Product of (1 - K_i) terms
    num_sources_fused: int = 0
    pairwise_conflicts: list = field(default_factory=list)


def get_hypotheses(mass: dict) -> list[str]:
    """
    Extract hypothesis keys (all keys except 'Omega') from a mass function.

    Args:
        mass: A mass function dict like {"K": 0.3, "NK": 0.2, "Omega": 0.5}.

    Returns:
        List of hypothesis keys, e.g. ["K", "NK"].
    """
    return [k for k in mass if k != "Omega"]


def validate_mass(mass: dict) -> dict:
    """
    Validate and re-normalize a mass function to ensure it sums to 1.0.

    Handles edge cases:
    - Negative values are clamped to 0
    - If total is 0, returns full ignorance (Omega = 1.0)
    - Small floating-point deviations are corrected

    Args:
        mass: Raw mass function dict.

    Returns:
        Validated and normalized mass function.
    """
    # Clamp negatives to zero
    cleaned = {k: max(0.0, v) for k, v in mass.items()}

    # Ensure Omega key exists
    if "Omega" not in cleaned:
        cleaned["Omega"] = 0.0

    total = sum(cleaned.values())

    if total == 0:
        # Full ignorance fallback
        result = {k: 0.0 for k in cleaned}
        result["Omega"] = 1.0
        return result

    # Normalize to sum to 1.0
    return {k: v / total for k, v in cleaned.items()}


def normalize_mass(mass: dict) -> dict:
    """
    Normalize a mass function so all values sum to exactly 1.0.
    Generalized: works with any set of hypotheses + Omega.

    Args:
        mass: Mass function dict, e.g. {"K": 0.3, "NK": 0.2, "Omega": 0.5}.

    Returns:
        Normalized mass function.
    """
    total = sum(mass.values())
    if total == 0:
        result = {k: 0.0 for k in mass}
        result["Omega"] = 1.0
        return result
    return {k: v / total for k, v in mass.items()}


def apply_discount(mass: dict, alpha: float) -> dict:
    """
    Apply Shafer's discounting operation based on source reliability.

    For a source with reliability α ∈ [0, 1]:
    - m'(A) = α · m(A) for each focal element A ⊂ Θ
    - m'(Ω) = α · m(Ω) + (1 - α)

    When α = 0, the source is completely unreliable → full ignorance.
    When α = 1, the source is fully reliable → mass unchanged.

    Args:
        mass: Original mass function.
        alpha: Reliability/confidence factor in [0, 1].

    Returns:
        Discounted mass function.
    """
    alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]

    discounted = {}
    for key, value in mass.items():
        if key == "Omega":
            discounted[key] = value * alpha + (1.0 - alpha)
        else:
            discounted[key] = value * alpha

    return normalize_mass(discounted)


def dempster_combine(m1: dict, m2: dict) -> tuple[dict, float]:
    """
    Combine two mass functions using Dempster's Rule of Combination.

    Generalized for arbitrary frames with singleton focal elements + Omega.

    For singletons Hi:
        m_fused(Hi) = [m1(Hi)·m2(Hi) + m1(Hi)·m2(Ω) + m1(Ω)·m2(Hi)] / (1 - K)

    For Omega (full frame):
        m_fused(Ω) = m1(Ω)·m2(Ω) / (1 - K)

    Conflict:
        K = Σ_{i≠j} m1(Hi)·m2(Hj) for all distinct singletons Hi, Hj

    Args:
        m1: First mass function.
        m2: Second mass function (must have same hypotheses as m1).

    Returns:
        Tuple of (combined mass function, conflict degree K).

    Raises:
        ValueError: If the two mass functions have different hypothesis sets.
    """
    hyp1 = set(get_hypotheses(m1))
    hyp2 = set(get_hypotheses(m2))
    if hyp1 != hyp2:
        raise ValueError(
            f"Mass functions have different hypothesis sets: {hyp1} vs {hyp2}"
        )

    hypotheses = list(hyp1)

    # Compute conflict: K = sum of m1(Hi) * m2(Hj) for Hi ≠ Hj
    conflict = 0.0
    for hi in hypotheses:
        for hj in hypotheses:
            if hi != hj:
                conflict += m1[hi] * m2[hj]

    # If conflict is total, return full ignorance
    if conflict >= 1.0:
        result = {h: 0.0 for h in hypotheses}
        result["Omega"] = 1.0
        return result, conflict

    normalization = 1.0 / (1.0 - conflict)

    # Compute fused mass for each singleton
    fused = {}
    for h in hypotheses:
        # Hi ∩ Hi = Hi (agreement on Hi)
        # Hi ∩ Ω = Hi (evidence + ignorance)
        # Ω ∩ Hi = Hi (ignorance + evidence)
        fused[h] = (
            m1[h] * m2[h]
            + m1[h] * m2["Omega"]
            + m1["Omega"] * m2[h]
        ) * normalization

    # Omega ∩ Omega = Omega (mutual ignorance)
    fused["Omega"] = m1["Omega"] * m2["Omega"] * normalization

    return normalize_mass(fused), conflict


def dempster_fuse_all(masses: list[dict]) -> tuple[dict, FusionMetrics]:
    """
    Sequentially fuse N mass functions using Dempster's Rule.

    Args:
        masses: List of mass functions to fuse (minimum 1).

    Returns:
        Tuple of (final fused mass, FusionMetrics with conflict tracking).

    Raises:
        ValueError: If the list is empty.
    """
    if not masses:
        raise ValueError("Cannot fuse empty list of mass functions.")

    if len(masses) == 1:
        metrics = FusionMetrics(num_sources_fused=1)
        return masses[0], metrics

    metrics = FusionMetrics()
    current_mass = masses[0]
    normalization_product = 1.0  # Track Π(1 - K_i)

    for i, next_mass in enumerate(masses[1:], start=1):
        current_mass, k = dempster_combine(current_mass, next_mass)
        metrics.pairwise_conflicts.append(k)
        metrics.max_pairwise_conflict = max(metrics.max_pairwise_conflict, k)
        normalization_product *= (1.0 - k)

    metrics.total_conflict_product = 1.0 - normalization_product
    metrics.num_sources_fused = len(masses)

    return current_mass, metrics


def compute_belief_plausibility(mass: dict) -> dict:
    """
    Compute Belief (Bel) and Plausibility (Pl) for each singleton hypothesis.

    For our simplified frame (singletons + Omega only):
    - Bel(Hi) = m(Hi)  (only the singleton itself supports Hi exactly)
    - Pl(Hi) = m(Hi) + m(Ω)  (Hi could be true if Ω is resolved to Hi)
    - Uncertainty interval: [Bel(Hi), Pl(Hi)]

    Args:
        mass: A fused/final mass function.

    Returns:
        Dict mapping each hypothesis to {"belief": float, "plausibility": float}.
    """
    hypotheses = get_hypotheses(mass)
    omega = mass.get("Omega", 0.0)

    result = {}
    for h in hypotheses:
        bel = mass[h]
        pl = mass[h] + omega  # Plausibility includes the undecided mass
        result[h] = {"belief": round(bel, 4), "plausibility": round(pl, 4)}

    return result


def decide(mass: dict, omega_threshold: float = 0.5) -> tuple[str, float]:
    """
    Make a decision from a mass function.

    Rules:
    1. If Omega > omega_threshold → "不確実" (uncertain)
    2. Otherwise → hypothesis with highest mass, with that mass as confidence

    Args:
        mass: A mass function dict.
        omega_threshold: Threshold above which Omega triggers an uncertain verdict.

    Returns:
        Tuple of (prediction string, confidence value).
    """
    hypotheses = get_hypotheses(mass)

    if mass.get("Omega", 0.0) > omega_threshold:
        return "不確実", round(mass["Omega"], 3)

    # Find hypothesis with highest mass
    best_h = max(hypotheses, key=lambda h: mass[h])
    return best_h, round(mass[best_h], 3)
