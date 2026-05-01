# -*- coding: utf-8 -*-
"""
Unit tests for the Dempster-Shafer Theory engine.

Validates all DST mathematical operations against known manual calculations.
"""

import pytest

from demographic_inference.dst_engine import (
    apply_discount,
    compute_belief_plausibility,
    decide,
    dempster_combine,
    dempster_fuse_all,
    get_hypotheses,
    normalize_mass,
    validate_mass,
)


# ─── normalize_mass ───────────────────────────────────────────────────

class TestNormalizeMass:
    def test_already_normalized(self):
        m = {"K": 0.3, "NK": 0.2, "Omega": 0.5}
        result = normalize_mass(m)
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_unnormalized(self):
        m = {"K": 0.6, "NK": 0.4, "Omega": 1.0}
        result = normalize_mass(m)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["K"] - 0.3) < 1e-9

    def test_all_zero(self):
        m = {"K": 0, "NK": 0, "Omega": 0}
        result = normalize_mass(m)
        assert result["Omega"] == 1.0
        assert result["K"] == 0.0

    def test_multi_hypothesis(self):
        """Test with age-like frame (7 elements)."""
        m = {"18_24": 0.2, "25_34": 0.3, "35_44": 0.1, "45_54": 0.0,
             "55_64": 0.0, "65_plus": 0.0, "Omega": 0.4}
        result = normalize_mass(m)
        assert abs(sum(result.values()) - 1.0) < 1e-9


# ─── validate_mass ────────────────────────────────────────────────────

class TestValidateMass:
    def test_negative_values_clamped(self):
        m = {"K": -0.1, "NK": 0.3, "Omega": 0.8}
        result = validate_mass(m)
        assert result["K"] == 0.0
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_missing_omega(self):
        m = {"K": 0.3, "NK": 0.7}
        result = validate_mass(m)
        assert "Omega" in result
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_all_zero(self):
        m = {"K": 0, "NK": 0, "Omega": 0}
        result = validate_mass(m)
        assert result["Omega"] == 1.0


# ─── apply_discount ───────────────────────────────────────────────────

class TestApplyDiscount:
    def test_full_confidence(self):
        """alpha=1.0 should leave mass unchanged."""
        m = {"K": 0.6, "NK": 0.2, "Omega": 0.2}
        result = apply_discount(m, alpha=1.0)
        assert abs(result["K"] - 0.6) < 1e-9
        assert abs(result["NK"] - 0.2) < 1e-9
        assert abs(result["Omega"] - 0.2) < 1e-9

    def test_zero_confidence(self):
        """alpha=0.0 should result in full ignorance."""
        m = {"K": 0.6, "NK": 0.2, "Omega": 0.2}
        result = apply_discount(m, alpha=0.0)
        assert abs(result["K"] - 0.0) < 1e-9
        assert abs(result["NK"] - 0.0) < 1e-9
        assert abs(result["Omega"] - 1.0) < 1e-9

    def test_partial_discount(self):
        """alpha=0.5 should shift mass toward Omega."""
        m = {"K": 0.8, "NK": 0.0, "Omega": 0.2}
        result = apply_discount(m, alpha=0.5)
        # m'(K) = 0.8*0.5 = 0.4
        # m'(NK) = 0.0*0.5 = 0.0
        # m'(Omega) = 0.2*0.5 + 0.5 = 0.6
        assert abs(result["K"] - 0.4) < 1e-9
        assert abs(result["NK"] - 0.0) < 1e-9
        assert abs(result["Omega"] - 0.6) < 1e-9

    def test_sum_to_one(self):
        m = {"K": 0.5, "NK": 0.3, "Omega": 0.2}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = apply_discount(m, alpha)
            assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_alpha_clamped(self):
        """Alpha values outside [0,1] should be clamped."""
        m = {"K": 0.5, "NK": 0.3, "Omega": 0.2}
        result_neg = apply_discount(m, alpha=-0.5)
        result_zero = apply_discount(m, alpha=0.0)
        assert result_neg == result_zero

    def test_multi_hypothesis_discount(self):
        """Discounting works for gender frame too."""
        m = {"M": 0.6, "F": 0.1, "Omega": 0.3}
        result = apply_discount(m, alpha=0.5)
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result["Omega"] > 0.3  # Omega should increase


# ─── dempster_combine ─────────────────────────────────────────────────

class TestDempsterCombine:
    def test_agreement_strengthens(self):
        """Two sources agreeing on K should produce higher K belief."""
        m1 = {"K": 0.6, "NK": 0.1, "Omega": 0.3}
        m2 = {"K": 0.7, "NK": 0.0, "Omega": 0.3}
        result, conflict = dempster_combine(m1, m2)
        assert result["K"] > max(m1["K"], m2["K"])
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_contradiction_produces_conflict(self):
        """Two sources disagreeing should produce conflict."""
        m1 = {"K": 0.8, "NK": 0.0, "Omega": 0.2}
        m2 = {"K": 0.0, "NK": 0.8, "Omega": 0.2}
        result, conflict = dempster_combine(m1, m2)
        assert conflict > 0.5
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_full_ignorance_neutral(self):
        """Combining with full ignorance should not change the other mass."""
        m1 = {"K": 0.5, "NK": 0.3, "Omega": 0.2}
        m2 = {"K": 0.0, "NK": 0.0, "Omega": 1.0}
        result, conflict = dempster_combine(m1, m2)
        assert conflict == 0.0
        assert abs(result["K"] - m1["K"]) < 1e-9
        assert abs(result["NK"] - m1["NK"]) < 1e-9

    def test_full_conflict_returns_ignorance(self):
        """Total conflict (K>=1) should return full ignorance."""
        m1 = {"K": 1.0, "NK": 0.0, "Omega": 0.0}
        m2 = {"K": 0.0, "NK": 1.0, "Omega": 0.0}
        result, conflict = dempster_combine(m1, m2)
        assert conflict >= 1.0
        assert result["Omega"] == 1.0

    def test_manual_calculation(self):
        """Verify against a hand-calculated example."""
        m1 = {"K": 0.4, "NK": 0.2, "Omega": 0.4}
        m2 = {"K": 0.3, "NK": 0.3, "Omega": 0.4}

        # Conflict: K = m1(K)*m2(NK) + m1(NK)*m2(K)
        # K = 0.4*0.3 + 0.2*0.3 = 0.12 + 0.06 = 0.18
        expected_conflict = 0.18

        # Numerators:
        # K_num = m1(K)*m2(K) + m1(K)*m2(Ω) + m1(Ω)*m2(K)
        #       = 0.4*0.3 + 0.4*0.4 + 0.4*0.3 = 0.12 + 0.16 + 0.12 = 0.40
        # NK_num = m1(NK)*m2(NK) + m1(NK)*m2(Ω) + m1(Ω)*m2(NK)
        #        = 0.2*0.3 + 0.2*0.4 + 0.4*0.3 = 0.06 + 0.08 + 0.12 = 0.26
        # Ω_num = m1(Ω)*m2(Ω) = 0.4*0.4 = 0.16

        # Normalized: divide by (1 - 0.18) = 0.82
        # K = 0.40/0.82 ≈ 0.4878
        # NK = 0.26/0.82 ≈ 0.3171
        # Ω = 0.16/0.82 ≈ 0.1951

        result, conflict = dempster_combine(m1, m2)

        assert abs(conflict - expected_conflict) < 1e-9
        assert abs(result["K"] - 0.40 / 0.82) < 1e-4
        assert abs(result["NK"] - 0.26 / 0.82) < 1e-4
        assert abs(result["Omega"] - 0.16 / 0.82) < 1e-4

    def test_mismatched_hypotheses_raises(self):
        """Different hypothesis sets should raise ValueError."""
        m1 = {"K": 0.5, "NK": 0.3, "Omega": 0.2}
        m2 = {"M": 0.5, "F": 0.3, "Omega": 0.2}
        with pytest.raises(ValueError):
            dempster_combine(m1, m2)

    def test_symmetric(self):
        """Dempster's rule should be commutative."""
        m1 = {"K": 0.4, "NK": 0.2, "Omega": 0.4}
        m2 = {"K": 0.3, "NK": 0.5, "Omega": 0.2}
        r1, k1 = dempster_combine(m1, m2)
        r2, k2 = dempster_combine(m2, m1)
        assert abs(k1 - k2) < 1e-9
        for key in r1:
            assert abs(r1[key] - r2[key]) < 1e-9


# ─── dempster_fuse_all ────────────────────────────────────────────────

class TestDempsterFuseAll:
    def test_single_mass(self):
        m = {"K": 0.5, "NK": 0.3, "Omega": 0.2}
        result, metrics = dempster_fuse_all([m])
        assert result == m
        assert metrics.num_sources_fused == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            dempster_fuse_all([])

    def test_multiple_fusion_tracks_conflict(self):
        masses = [
            {"K": 0.4, "NK": 0.1, "Omega": 0.5},
            {"K": 0.3, "NK": 0.2, "Omega": 0.5},
            {"K": 0.5, "NK": 0.0, "Omega": 0.5},
        ]
        result, metrics = dempster_fuse_all(masses)
        assert metrics.num_sources_fused == 3
        assert len(metrics.pairwise_conflicts) == 2
        assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_four_sources_like_pipeline(self):
        """Simulate the real pipeline: 4 discounted sources."""
        mA = {"K": 0.2, "NK": 0.1, "Omega": 0.7}
        mB = {"K": 0.0, "NK": 0.0, "Omega": 1.0}
        mC = {"K": 0.15, "NK": 0.0, "Omega": 0.85}
        mD = {"K": 0.0, "NK": 0.0, "Omega": 1.0}
        result, metrics = dempster_fuse_all([mA, mB, mC, mD])
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert result["K"] > 0  # Should have some K belief


# ─── compute_belief_plausibility ──────────────────────────────────────

class TestBeliefPlausibility:
    def test_basic(self):
        m = {"K": 0.4, "NK": 0.3, "Omega": 0.3}
        bp = compute_belief_plausibility(m)
        assert abs(bp["K"]["belief"] - 0.4) < 1e-4
        assert abs(bp["K"]["plausibility"] - 0.7) < 1e-4
        assert abs(bp["NK"]["belief"] - 0.3) < 1e-4
        assert abs(bp["NK"]["plausibility"] - 0.6) < 1e-4

    def test_full_ignorance(self):
        m = {"K": 0.0, "NK": 0.0, "Omega": 1.0}
        bp = compute_belief_plausibility(m)
        assert bp["K"]["belief"] == 0.0
        assert bp["K"]["plausibility"] == 1.0


# ─── decide ───────────────────────────────────────────────────────────

class TestDecide:
    def test_certain_k(self):
        m = {"K": 0.7, "NK": 0.2, "Omega": 0.1}
        pred, conf = decide(m)
        assert pred == "K"
        assert abs(conf - 0.7) < 1e-3

    def test_uncertain(self):
        m = {"K": 0.1, "NK": 0.1, "Omega": 0.8}
        pred, conf = decide(m)
        assert pred == "不確実"
        assert abs(conf - 0.8) < 1e-3

    def test_threshold_edge(self):
        m = {"K": 0.3, "NK": 0.2, "Omega": 0.5}
        pred, conf = decide(m, omega_threshold=0.5)
        assert pred == "K"  # Omega == threshold, not strictly greater
