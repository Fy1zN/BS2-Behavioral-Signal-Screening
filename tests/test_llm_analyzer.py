"""
Unit tests for Google Gemini API integration and error handling with mocks.
Ensures zero external network dependencies during unit testing.
"""

from unittest.mock import MagicMock, patch
import pytest
from src.llm_analyzer import (
    GeminiAnalyzer,
    GeminiAuthError,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiRateLimitError,
    GeminiResponseValidationError,
    GeminiTimeoutError,
    build_analysis_prompt,
)
from src.models import NLPFeatures
from src.utils import extract_json_from_text


@pytest.fixture
def sample_nlp_features():
    return NLPFeatures(
        character_count=40,
        word_count=8,
        sentence_count=1,
        avg_sentence_length=8.0,
        number_count=1,
        exclamation_count=1,
        question_count=0,
        uppercase_word_count=0,
        uppercase_ratio=0.05,
        sentiment_score=-0.1,
        sentiment_label="NEUTRAL",
        named_entities=[],
        scarcity_indicators=["only 2 left"],
        urgency_indicators=["act fast"],
        fomo_indicators=[],
        social_proof_indicators=[],
        confirmshaming_indicators=[],
        forced_continuity_indicators=[],
        total_indicator_count=2,
        nlp_heuristic_score=40.0,
    )


def test_build_analysis_prompt(sample_nlp_features):
    """Verify prompt replaces placeholders with text and compact NLP features."""
    text = "Only 2 rooms left! Act fast."
    prompt = build_analysis_prompt(text, sample_nlp_features)
    assert text in prompt
    assert "SCARCITY" in prompt
    assert "nlp_heuristic_score" in prompt


def test_extract_json_direct():
    """Verify extraction of standard raw JSON."""
    raw = '{"is_dark_pattern": true, "primary_category": "SCARCITY"}'
    parsed = extract_json_from_text(raw)
    assert parsed["primary_category"] == "SCARCITY"


def test_extract_json_markdown_fence():
    """Verify extraction of JSON wrapped inside markdown code fences."""
    raw = """Here is the analysis:
```json
{
  "is_dark_pattern": true,
  "primary_category": "URGENCY"
}
```
Done."""
    parsed = extract_json_from_text(raw)
    assert parsed["primary_category"] == "URGENCY"


def test_extract_json_empty_raises():
    """Empty or unparseable text must raise ValueError."""
    with pytest.raises(ValueError):
        extract_json_from_text("")


def test_gemini_missing_api_key_raises():
    """Missing API key must raise GeminiConfigError without attempting network calls."""
    with patch("src.llm_analyzer.get_gemini_api_key", side_effect=ValueError("Gemini API key is not configured")):
        with pytest.raises(GeminiConfigError) as exc_info:
            GeminiAnalyzer(api_key="")
        assert "not configured" in str(exc_info.value)


@patch("src.llm_analyzer.get_gemini_api_key", return_value="fake_key_123")
@patch("src.llm_analyzer.resolve_gemini_model", return_value="gemini-2.5-flash")
def test_gemini_analyzer_successful_mock(mock_model, mock_key, sample_nlp_features):
    """Successful mock response validates into LLMAnalysis."""
    valid_json_response = """
    {
      "is_dark_pattern": true,
      "primary_category": "SCARCITY",
      "secondary_categories": ["URGENCY"],
      "manipulation_score": 85.0,
      "severity": "CRITICAL",
      "confidence": 0.92,
      "detected_phrases": ["Only 2 left", "Act fast"],
      "psychological_mechanisms": ["Fear of Missing Out"],
      "explanation": "Creates pressure using scarcity.",
      "recommendation": "Display stock neutrally."
    }
    """
    analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.5-flash")
    with patch.object(analyzer, "_call_gemini_api", return_value=valid_json_response):
        result = analyzer.analyze("Only 2 left! Act fast.", sample_nlp_features)
        assert result.is_dark_pattern is True
        assert result.primary_category == "SCARCITY"
        assert result.manipulation_score == 85.0
        assert result.confidence == 0.92


@patch("src.llm_analyzer.get_gemini_api_key", return_value="fake_key_123")
@patch("src.llm_analyzer.resolve_gemini_model", return_value="gemini-2.5-flash")
def test_gemini_analyzer_recovery_mechanism(mock_model, mock_key, sample_nlp_features):
    """When first response is malformed, single recovery attempt succeeds."""
    malformed_first = "Invalid non-json response text"
    valid_recovery = """
    {
      "is_dark_pattern": false,
      "primary_category": "NONE",
      "secondary_categories": [],
      "manipulation_score": 10.0,
      "severity": "NONE",
      "confidence": 0.95,
      "detected_phrases": [],
      "psychological_mechanisms": [],
      "explanation": "Standard neutral order total statement.",
      "recommendation": "Maintain clear factual receipts."
    }
    """
    analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.5-flash")
    with patch.object(analyzer, "_call_gemini_api", side_effect=[malformed_first, valid_recovery]) as mock_call:
        result = analyzer.analyze("Your order total is Rs 999.", sample_nlp_features)
        assert result.primary_category == "NONE"
        assert result.manipulation_score == 10.0
        assert mock_call.call_count == 2


@patch("src.llm_analyzer.get_gemini_api_key", return_value="fake_key_123")
@patch("src.llm_analyzer.resolve_gemini_model", return_value="gemini-2.5-flash")
def test_gemini_analyzer_recovery_failure_raises(mock_model, mock_key, sample_nlp_features):
    """When both primary and recovery attempts fail, raise GeminiResponseValidationError."""
    malformed = "Still completely unparseable text"
    analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.5-flash")
    with patch.object(analyzer, "_call_gemini_api", return_value=malformed):
        with pytest.raises(GeminiResponseValidationError):
            analyzer.analyze("Test input", sample_nlp_features)


@patch("src.llm_analyzer.get_gemini_api_key", return_value="fake_key_123")
@patch("src.llm_analyzer.resolve_gemini_model", return_value="gemini-2.5-flash")
def test_gemini_error_classification(mock_model, mock_key):
    """Verify API error messages map to custom exceptions."""
    analyzer = GeminiAnalyzer(api_key="fake_key", model="gemini-2.5-flash")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 Resource exhausted: quota exceeded")
    analyzer._client = mock_client

    with pytest.raises(GeminiRateLimitError):
        analyzer._call_gemini_api("Test prompt")

    mock_client.models.generate_content.side_effect = Exception("401 API_KEY_INVALID: bad credentials")
    with pytest.raises(GeminiAuthError):
        analyzer._call_gemini_api("Test prompt")

    mock_client.models.generate_content.side_effect = Exception("Deadline exceeded: timeout reached")
    with pytest.raises(GeminiTimeoutError):
        analyzer._call_gemini_api("Test prompt")
