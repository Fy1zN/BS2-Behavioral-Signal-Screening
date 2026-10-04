"""
Unit tests for NLP feature extraction, linguistic indicators, sentiment, and NER.
"""

from src.nlp_features import (
    analyze_sentiment,
    extract_named_entities,
    extract_nlp_features,
    find_indicator_matches,
    INDICATOR_PATTERNS,
)


def test_linguistic_indicator_scarcity():
    """Verify scarcity indicators are captured accurately."""
    text = "Only 2 rooms left! Limited stock remaining."
    features = extract_nlp_features(text)
    assert len(features.scarcity_indicators) > 0
    assert any("2 rooms left" in match.lower() or "only 2" in match.lower() for match in features.scarcity_indicators)


def test_linguistic_indicator_urgency():
    """Verify urgency indicators are captured accurately."""
    text = "Hurry! Act now, sale ends today only!"
    features = extract_nlp_features(text)
    assert len(features.urgency_indicators) > 0
    assert any("hurry" in match.lower() for match in features.urgency_indicators)
    assert any("act now" in match.lower() for match in features.urgency_indicators)


def test_linguistic_indicator_confirmshaming():
    """Verify confirmshaming patterns like 'No thanks, I don't want'."""
    text = "No thanks, I don't want to save money and prefer paying full price."
    features = extract_nlp_features(text)
    assert len(features.confirmshaming_indicators) > 0
    assert any("no thanks" in match.lower() for match in features.confirmshaming_indicators)


def test_linguistic_indicator_forced_continuity():
    """Verify forced continuity triggers on auto-renew and trial language."""
    text = "Start your free trial now. Subscription automatically renews for $29/mo."
    features = extract_nlp_features(text)
    assert len(features.forced_continuity_indicators) > 0
    assert any("free trial" in match.lower() for match in features.forced_continuity_indicators)
    assert any("automatically renew" in match.lower() for match in features.forced_continuity_indicators)


def test_linguistic_indicator_social_proof():
    """Verify social proof manipulation matches viewer counts."""
    text = "47 people are viewing this product right now! Popular choice."
    features = extract_nlp_features(text)
    assert len(features.social_proof_indicators) > 0
    assert any("people are viewing" in match.lower() for match in features.social_proof_indicators)


def test_sentiment_analysis():
    """Test sentiment polarity calculation and label output."""
    pos_score, pos_label = analyze_sentiment("Great product, transparent pricing, helpful support!")
    assert pos_score > 0
    assert pos_label == "POSITIVE"

    neg_score, neg_label = analyze_sentiment("I hate this and risk getting infected by malware.")
    assert neg_score < 0
    assert neg_label == "NEGATIVE"


def test_named_entity_extraction():
    """Verify currency and numeric entities are detected."""
    text = "Your room fee is ₹2,499 payable on October 15th."
    entities = extract_named_entities(text)
    assert len(entities) > 0
    entity_texts = [e.text for e in entities]
    assert any("2,499" in t or "2499" in t for t in entity_texts)


def test_nlp_features_normal_text():
    """Normal neutral text should yield 0 or minimal dark pattern indicators."""
    text = "Your order total is ₹999 including taxes and delivery."
    features = extract_nlp_features(text)
    assert features.total_indicator_count == 0
    assert features.nlp_heuristic_score < 20.0
