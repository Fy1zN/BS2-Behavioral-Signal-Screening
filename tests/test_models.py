"""
Unit tests for Pydantic data models and strict validation.
"""

import pytest
from pydantic import ValidationError
from src.models import DarkPatternCategory, FinalAnalysis, LLMAnalysis, NLPFeatures, SeverityLevel


def test_llm_analysis_valid():
    """Valid LLM analysis payload should parse cleanly."""
    valid_data = {
        "is_dark_pattern": True,
        "primary_category": "SCARCITY",
        "secondary_categories": ["URGENCY"],
        "manipulation_score": 91.0,
        "severity": "CRITICAL",
        "confidence": 0.94,
        "detected_phrases": ["Only 2 rooms left", "Book now"],
        "psychological_mechanisms": ["Fear of Missing Out", "Loss Aversion"],
        "explanation": "Emphasizes limited availability to create immediate action.",
        "recommendation": "Present availability neutrally.",
    }
    model = LLMAnalysis(**valid_data)
    assert model.is_dark_pattern is True
    assert model.primary_category == "SCARCITY"
    assert model.manipulation_score == 91.0
    assert model.severity == "CRITICAL"
    assert model.confidence == 0.94


def test_llm_analysis_invalid_category():
    """Unsupported dark pattern category must raise ValidationError."""
    invalid_data = {
        "is_dark_pattern": True,
        "primary_category": "NOT_A_REAL_CATEGORY",
        "secondary_categories": [],
        "manipulation_score": 50,
        "severity": "MEDIUM",
        "confidence": 0.8,
        "detected_phrases": [],
        "psychological_mechanisms": [],
        "explanation": "Valid explanation text.",
        "recommendation": "Valid recommendation text.",
    }
    with pytest.raises(ValidationError):
        LLMAnalysis(**invalid_data)


def test_llm_analysis_invalid_score():
    """Score outside 0-100 range must raise ValidationError."""
    invalid_data = {
        "is_dark_pattern": True,
        "primary_category": "SCARCITY",
        "secondary_categories": [],
        "manipulation_score": 150.0,  # Invalid
        "severity": "CRITICAL",
        "confidence": 0.9,
        "detected_phrases": [],
        "psychological_mechanisms": [],
        "explanation": "Valid explanation text.",
        "recommendation": "Valid recommendation text.",
    }
    with pytest.raises(ValidationError):
        LLMAnalysis(**invalid_data)


def test_llm_analysis_invalid_confidence():
    """Confidence outside 0.0-1.0 range must raise ValidationError."""
    invalid_data = {
        "is_dark_pattern": True,
        "primary_category": "SCARCITY",
        "secondary_categories": [],
        "manipulation_score": 75.0,
        "severity": "HIGH",
        "confidence": 1.5,  # Invalid
        "detected_phrases": [],
        "psychological_mechanisms": [],
        "explanation": "Valid explanation text.",
        "recommendation": "Valid recommendation text.",
    }
    with pytest.raises(ValidationError):
        LLMAnalysis(**invalid_data)


def test_llm_analysis_invalid_severity():
    """Invalid severity level must raise ValidationError."""
    invalid_data = {
        "is_dark_pattern": True,
        "primary_category": "SCARCITY",
        "secondary_categories": [],
        "manipulation_score": 75.0,
        "severity": "SUPER_EXTREME",  # Invalid
        "confidence": 0.8,
        "detected_phrases": [],
        "psychological_mechanisms": [],
        "explanation": "Valid explanation text.",
        "recommendation": "Valid recommendation text.",
    }
    with pytest.raises(ValidationError):
        LLMAnalysis(**invalid_data)


def test_llm_analysis_empty_explanation_fails():
    """Explanation shorter than 5 chars must fail validation."""
    invalid_data = {
        "is_dark_pattern": True,
        "primary_category": "SCARCITY",
        "secondary_categories": [],
        "manipulation_score": 75.0,
        "severity": "HIGH",
        "confidence": 0.8,
        "detected_phrases": [],
        "psychological_mechanisms": [],
        "explanation": "bad",
        "recommendation": "Valid recommendation text.",
    }
    with pytest.raises(ValidationError):
        LLMAnalysis(**invalid_data)
