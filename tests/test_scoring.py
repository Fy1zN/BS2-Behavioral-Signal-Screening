"""
Unit tests for hybrid scoring, boundary conditions, and severity level mapping.
"""

from src.models import LLMAnalysis, NLPFeatures
from src.scoring import (
    build_final_analysis,
    compute_hybrid_score,
    score_to_severity,
)


def test_severity_boundary_mappings():
    """Verify all severity threshold boundaries are strictly mapped."""
    # 0 - 19: NONE
    assert score_to_severity(0.0) == "NONE"
    assert score_to_severity(10.0) == "NONE"
    assert score_to_severity(19.9) == "NONE"

    # 20 - 39: LOW
    assert score_to_severity(20.0) == "LOW"
    assert score_to_severity(30.0) == "LOW"
    assert score_to_severity(39.9) == "LOW"

    # 40 - 59: MEDIUM
    assert score_to_severity(40.0) == "MEDIUM"
    assert score_to_severity(50.0) == "MEDIUM"
    assert score_to_severity(59.9) == "MEDIUM"

    # 60 - 79: HIGH
    assert score_to_severity(60.0) == "HIGH"
    assert score_to_severity(70.0) == "HIGH"
    assert score_to_severity(79.9) == "HIGH"

    # 80 - 100: CRITICAL
    assert score_to_severity(80.0) == "CRITICAL"
    assert score_to_severity(95.0) == "CRITICAL"
    assert score_to_severity(100.0) == "CRITICAL"


def test_hybrid_score_calculation():
    """Verify formula: 0.70 * LLM + 0.30 * NLP."""
    # 0.70 * 90 + 0.30 * 50 = 63.0 + 15.0 = 78.0
    score = compute_hybrid_score(llm_score=90.0, nlp_score=50.0, llm_weight=0.70, nlp_weight=0.30)
    assert score == 78.0


def test_hybrid_score_extremes():
    """Verify clamping at 0 and 100."""
    assert compute_hybrid_score(0.0, 0.0) == 0.0
    assert compute_hybrid_score(100.0, 100.0) == 100.0


def test_build_final_analysis_integration():
    """Verify build_final_analysis produces a valid FinalAnalysis object."""
    nlp_feats = NLPFeatures(
        character_count=35,
        word_count=7,
        sentence_count=2,
        avg_sentence_length=3.5,
        number_count=1,
        exclamation_count=1,
        question_count=0,
        uppercase_word_count=0,
        uppercase_ratio=0.05,
        sentiment_score=-0.2,
        sentiment_label="NEGATIVE",
        named_entities=[],
        scarcity_indicators=["2 rooms left"],
        urgency_indicators=["book now"],
        fomo_indicators=[],
        social_proof_indicators=[],
        confirmshaming_indicators=[],
        forced_continuity_indicators=[],
        total_indicator_count=2,
        nlp_heuristic_score=41.0,
    )
    llm_output = LLMAnalysis(
        is_dark_pattern=True,
        primary_category="SCARCITY",
        secondary_categories=["URGENCY"],
        manipulation_score=90.0,
        severity="CRITICAL",
        confidence=0.95,
        detected_phrases=["2 rooms left", "book now"],
        psychological_mechanisms=["Fear of Missing Out"],
        explanation="Text exploits scarcity to trigger urgency.",
        recommendation="Display neutral stock levels.",
    )

    final = build_final_analysis(
        input_text="Only 2 rooms left! Book now.",
        nlp_features=nlp_feats,
        llm_analysis=llm_output,
        model_name="gemini-2.5-flash",
    )

    # 0.70 * 90 + 0.30 * 41 = 63.0 + 12.3 = 75.3
    assert final.final_score == 75.3
    assert final.severity == "HIGH"
    assert final.is_manipulative is True
    assert final.primary_category == "SCARCITY"
    assert "2 rooms left" in final.detected_evidence
