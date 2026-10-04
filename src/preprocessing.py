"""
NLP Preprocessing module for BS².
Handles input validation, whitespace normalization, sentence segmentation, and tokenization
while preserving punctuation and original text fidelity.
"""

import re
from typing import Dict, List, Optional, Tuple


def validate_input_text(text: Optional[str], max_length: int = 6000) -> Tuple[bool, Optional[str]]:
    """
    Validate user input text against emptiness, whitespace-only, and maximum length constraints.
    Returns (is_valid, error_message).
    """
    if text is None:
        return False, "Input text is required."

    stripped = text.strip()
    if not stripped:
        return False, "Input text cannot be empty or whitespace only."

    if len(text) > max_length:
        return (
            False,
            f"Input exceeds maximum allowed length of {max_length} characters (received {len(text)} characters).",
        )

    return True, None


def normalize_whitespace(text: str) -> str:
    """
    Normalizes excessive internal whitespace while preserving line structure and punctuation.
    """
    # Replace tabs and irregular whitespace sequences on each line with a single space
    lines = text.splitlines()
    normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    # Filter multiple consecutive blank lines down to a single blank line
    output = []
    blank = False
    for line in normalized_lines:
        if line:
            output.append(line)
            blank = False
        elif not blank:
            output.append("")
            blank = True
    return "\n".join(output).strip()


def segment_sentences(text: str) -> List[str]:
    """
    Segment text into sentences using regex boundary detection that preserves punctuation.
    """
    # Split on sentence terminals (. ! ?) followed by whitespace or quotes
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'₹$])", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences and text.strip():
        sentences = [text.strip()]
    return sentences


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words while keeping contractions and alphanumeric units intact.
    """
    return re.findall(r"\b[\w'-]+\b", text)


def preprocess_text(text: str, max_length: int = 6000) -> Dict[str, any]:
    """
    Validate, normalize, and extract basic structural token units.
    """
    is_valid, error = validate_input_text(text, max_length=max_length)
    if not is_valid:
        raise ValueError(error)

    normalized = normalize_whitespace(text)
    sentences = segment_sentences(normalized)
    tokens = tokenize_words(normalized)

    return {
        "original_text": text,
        "normalized_text": normalized,
        "sentences": sentences,
        "tokens": tokens,
        "char_count": len(text),
        "word_count": len(tokens),
        "sentence_count": len(sentences),
    }
