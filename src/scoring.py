"""
Scoring module for BS².
Computes hybrid manipulation scores combining NLP deterministic indicators
with Google Gemini contextual semantic scoring, mapping to explainable severity tiers.
"""

from typing import Any, Dict, List, Optional
from src.models import (
    DarkPatternCategory,
    FinalAnalysis,
    LLMAnalysis,
    NLPFeatures,
    SeverityLevel,
)
from src.utils import load_config

DEFAULT_SEVERITY_THRESHOLDS = {
    "NONE": (0.0, 19.99),
    "LOW": (20.0, 39.99),
    "MEDIUM": (40.0, 59.99),
    "HIGH": (60.0, 79.99),
    "CRITICAL": (80.0, 100.0),
}


def score_to_severity(score: float, thresholds: Optional[Dict[str, Any]] = None) -> SeverityLevel:
    """
    Map a numerical score (0-100) to an explainable SeverityLevel.
    0-19: NONE
    20-39: LOW
    40-59: MEDIUM
    60-79: HIGH
    80-100: CRITICAL
    """
    s = max(0.0, min(100.0, float(score)))

    if s < 20.0:
        return "NONE"
    elif s < 40.0:
        return "LOW"
    elif s < 60.0:
        return "MEDIUM"
    elif s < 80.0:
        return "HIGH"
    else:
        return "CRITICAL"


def compute_hybrid_score(
    llm_score: float,
    nlp_score: float,
    llm_weight: float = 0.70,
    nlp_weight: float = 0.30,
) -> float:
    """
    Calculate the combined hybrid manipulation score:
    final_score = (llm_weight * llm_score) + (nlp_weight * nlp_score)
    Clamped strictly to [0.0, 100.0].
    """
    # Normalize weights if sum != 1.0
    total_w = llm_weight + nlp_weight
    if total_w <= 0:
        w_llm, w_nlp = 0.70, 0.30
    else:
        w_llm = llm_weight / total_w
        w_nlp = nlp_weight / total_w

    raw_final = (w_llm * float(llm_score)) + (w_nlp * float(nlp_score))
    return round(max(0.0, min(100.0, raw_final)), 2)


def build_final_analysis(
    input_text: str,
    nlp_features: NLPFeatures,
    llm_analysis: LLMAnalysis,
    model_name: str = "gemini-2.5-flash",
    config: Optional[Dict[str, Any]] = None,
) -> FinalAnalysis:
    """
    Assemble the complete, validated FinalAnalysis artifact integrating NLP metrics,
    Google Gemini semantic classification, hybrid scoring, and explainable recommendations.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    analysis_cfg = config.get("analysis", {})
    llm_weight = float(analysis_cfg.get("llm_weight", 0.70))
    nlp_weight = float(analysis_cfg.get("nlp_weight", 0.30))

    nlp_score = nlp_features.nlp_heuristic_score
    llm_score = llm_analysis.manipulation_score

    final_score = compute_hybrid_score(
        llm_score=llm_score,
        nlp_score=nlp_score,
        llm_weight=llm_weight,
        nlp_weight=nlp_weight,
    )

    severity = score_to_severity(final_score)
    is_manipulative = final_score >= 35.0 or (
        llm_analysis.is_dark_pattern and llm_analysis.primary_category != "NONE"
    )

    # Collect combined detected evidence
    evidence = list(llm_analysis.detected_phrases)
    # Add unique NLP matches if not already represented
    for ind_list in [
        nlp_features.scarcity_indicators,
        nlp_features.urgency_indicators,
        nlp_features.fomo_indicators,
        nlp_features.social_proof_indicators,
        nlp_features.confirmshaming_indicators,
        nlp_features.forced_continuity_indicators,
    ]:
        for item in ind_list:
            if item not in evidence and any(item.lower() in phrase.lower() for phrase in evidence):
                continue
            if item not in evidence:
                evidence.append(item)

    return FinalAnalysis(
        input_text=input_text,
        nlp_features=nlp_features,
        llm_analysis=llm_analysis,
        nlp_score=nlp_score,
        llm_score=llm_score,
        final_score=final_score,
        severity=severity,
        is_manipulative=is_manipulative,
        primary_category=llm_analysis.primary_category,
        secondary_categories=llm_analysis.secondary_categories,
        detected_evidence=evidence,
        psychological_mechanisms=llm_analysis.psychological_mechanisms,
        explanation=llm_analysis.explanation,
        recommendation=llm_analysis.recommendation,
        weights_used={"llm_weight": llm_weight, "nlp_weight": nlp_weight},
        model_used=model_name,
    )
