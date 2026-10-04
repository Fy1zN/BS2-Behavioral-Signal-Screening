"""
Unit tests for NLP preprocessing and input validation.
"""

import pytest
from src.preprocessing import (
    normalize_whitespace,
    preprocess_text,
    segment_sentences,
    tokenize_words,
    validate_input_text,
)


def test_validate_input_empty():
    """Empty string must be rejected."""
    is_valid, msg = validate_input_text("")
    assert not is_valid
    assert "empty" in msg.lower()


def test_validate_input_none():
    """None input must be rejected."""
    is_valid, msg = validate_input_text(None)
    assert not is_valid
    assert "required" in msg.lower()


def test_validate_input_whitespace_only():
    """Whitespace-only input must be rejected."""
    is_valid, msg = validate_input_text("   \n\t  ")
    assert not is_valid
    assert "whitespace" in msg.lower()


def test_validate_input_oversized():
    """Input exceeding max length must be rejected."""
    oversized = "a" * 6001
    is_valid, msg = validate_input_text(oversized, max_length=6000)
    assert not is_valid
    assert "exceeds" in msg.lower()


def test_validate_input_valid():
    """Standard valid text must pass."""
    is_valid, msg = validate_input_text("Only 2 rooms left! Book now.", max_length=6000)
    assert is_valid
    assert msg is None


def test_normalize_whitespace():
    """Internal multiple spaces and extra blank lines should be normalized."""
    raw = "Only   2   rooms  left!\n\n\n\nBook   now."
    normalized = normalize_whitespace(raw)
    assert "Only 2 rooms left!" in normalized
    assert "   " not in normalized


def test_sentence_segmentation():
    """Sentences ending in ! . ? should be segmented while keeping punctuation."""
    text = "Only 2 rooms left! Book now before someone else takes it. Are you sure?"
    sents = segment_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Only 2 rooms left!"
    assert sents[1] == "Book now before someone else takes it."
    assert sents[2] == "Are you sure?"


def test_word_tokenization():
    """Tokens should extract words and handle contractions."""
    text = "Don't miss today's 50% discount!"
    tokens = tokenize_words(text)
    assert "Don't" in tokens or "dont" in [t.lower() for t in tokens]
    assert "discount" in tokens


def test_preprocess_text_pipeline():
    """Complete preprocessing pipeline returns dictionary with stats."""
    text = "Hurry! Only 3 left."
    result = preprocess_text(text)
    assert result["word_count"] == 4
    assert result["sentence_count"] == 2
    assert result["char_count"] == len(text)
    assert len(result["sentences"]) == 2


def test_preprocess_text_raises_on_invalid():
    """Preprocessing should raise ValueError on invalid input."""
    with pytest.raises(ValueError):
        preprocess_text("   ")
