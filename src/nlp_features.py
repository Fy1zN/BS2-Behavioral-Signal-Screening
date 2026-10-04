"""
NLP Feature Extraction module for BS².
Extracts statistical metrics, sentiment polarity, named entities (spaCy),
and linguistic pattern indicators across dark pattern categories.
"""

import re
from typing import Dict, List, Optional, Tuple
from src.models import NamedEntityItem, NLPFeatures
from src.preprocessing import preprocess_text

# Regex patterns for linguistic dark-pattern indicators
INDICATOR_PATTERNS = {
    "scarcity": [
        r"\bonly\s+\d+\b",
        r"\bonly\s+(?:a\s+)?few\b",
        r"\blimited\s+(?:stock|edition|time|availability|rooms?|seats?|tickets?)\b",
        r"\b(?:rooms?|seats?|items?|tickets?|pieces?)\s+left\b",
        r"\balmost\s+sold\s+out\b",
        r"\blow\s+stock\b",
        r"\blast\s+(?:chance|room|item|seat|spot)\b",
        r"\bremaining\b",
        r"\bexclusive\b",
    ],
    "urgency": [
        r"\bact\s+(?:now|fast)\b",
        r"\bhurry\b",
        r"\bbook\s+now\b",
        r"\border\s+now\b",
        r"\bbuy\s+now\b",
        r"\bcheckout\s+immediately\b",
        r"\bimmediately\b",
        r"\btoday\s+only\b",
        r"\bends?\s+(?:soon|today|tonight|in\s+\d+)\b",
        r"\bexpires?\b",
        r"\blast\s+chance\b",
        r"\bflash\s+sale\b",
        r"\bcountdown\b",
        r"\bbefore\s+midnight\b",
    ],
    "fomo": [
        r"\bdon'?t\s+miss\s+out\b",
        r"\bdon'?t\s+miss\b",
        r"\bbefore\s+it'?s\s+too\s+late\b",
        r"\bsomeone\s+else\s+will\s+take\b",
        r"\bbefore\s+someone\s+else\b",
        r"\blose\s+your\b",
        r"\bmiss\s+out\b",
    ],
    "social_proof": [
        r"\b\d+\s+people\s+are\s+viewing\b",
        r"\bpeople\s+are\s+viewing\b",
        r"\bcustomers?\s+(?:bought|booked|purchased)\b",
        r"\bpopular\b",
        r"\btrending\b",
        r"\bmost\s+users\b",
        r"\b\d+%\s+of\s+users\b",
        r"\bhigh\s+demand\b",
        r"\bin\s+high\s+demand\b",
    ],
    "confirmshaming": [
        r"\bno\s+thanks\b",
        r"\bi\s+don'?t\s+want\s+(?:to\s+save|discounts?|money|deals?)\b",
        r"\bi\s+hate\s+saving\b",
        r"\bcontinue\s+without\b",
        r"\bremain\s+unprotected\b",
        r"\brisk\s+my\b",
        r"\bprefer\s+paying\s+full\b",
        r"\bno,\s+i\s+prefer\b",
        r"\bno,\s+i\s+choose\b",
    ],
    "forced_continuity": [
        r"\bfree\s+trial\b",
        r"\bautomatically\s+renews?\b",
        r"\bauto-?renews?\b",
        r"\bauto-?billing\b",
        r"\bcharged\s+after\s+(?:the\s+)?trial\b",
        r"\brecurring\s+(?:billing|charge|subscription)\b",
        r"\brenews\s+automatically\b",
        r"\bwithout\s+prior\s+notice\b",
    ],
}

# Lazy-loaded spaCy NLP model and NLTK VADER analyzer
_SPACY_NLP = None
_VADER_ANALYZER = None


def get_spacy_nlp():
    """Lazy load spaCy model with graceful fallback."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            # If model not downloaded, download or create blank
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                _SPACY_NLP = spacy.load("en_core_web_sm")
            except Exception:
                _SPACY_NLP = spacy.blank("en")
                if "sentencizer" not in _SPACY_NLP.pipe_names:
                    _SPACY_NLP.add_pipe("sentencizer")
    except ImportError:
        _SPACY_NLP = None

    return _SPACY_NLP


def get_vader_analyzer():
    """Lazy load NLTK SentimentIntensityAnalyzer with lexicon verification."""
    global _VADER_ANALYZER
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER

    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
    except Exception:
        _VADER_ANALYZER = None

    return _VADER_ANALYZER


def analyze_sentiment(text: str) -> Tuple[float, str]:
    """
    Calculate sentiment polarity and qualitative label.
    Label is POSITIVE, NEUTRAL, or NEGATIVE.
    """
    analyzer = get_vader_analyzer()
    if analyzer is not None:
        scores = analyzer.polarity_scores(text)
        compound = float(scores["compound"])
        if compound >= 0.05:
            label = "POSITIVE"
        elif compound <= -0.05:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"
        return round(compound, 3), label

    # Lightweight rule-based fallback if NLTK is not initialized
    positive_words = {"save", "discount", "free", "best", "exclusive", "popular", "great", "reward"}
    negative_words = {"hate", "risk", "lose", "unprotected", "infected", "penalty", "loss", "danger"}

    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    pos_count = len(tokens.intersection(positive_words))
    neg_count = len(tokens.intersection(negative_words))

    if pos_count > neg_count:
        return 0.3, "POSITIVE"
    elif neg_count > pos_count:
        return -0.3, "NEGATIVE"
    return 0.0, "NEUTRAL"


def extract_named_entities(text: str) -> List[NamedEntityItem]:
    """
    Extract relevant named entities using spaCy (MONEY, DATE, CARDINAL, ORG, PERSON, GPE, PRODUCT).
    """
    nlp = get_spacy_nlp()
    target_labels = {"MONEY", "DATE", "CARDINAL", "ORG", "PERSON", "GPE", "PRODUCT", "TIME", "PERCENT"}
    entities: List[NamedEntityItem] = []

    if nlp is not None and hasattr(nlp, "pipe_names") and "ner" in nlp.pipe_names:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in target_labels:
                entities.append(NamedEntityItem(text=ent.text, label=ent.label_))
        return entities

    # Fallback regex extraction for MONEY, DATES/TIMES, and NUMBERS
    money_matches = re.findall(r"[\$₹€£]\s*\d+(?:[.,]\d+)?", text)
    for m in money_matches:
        entities.append(NamedEntityItem(text=m, label="MONEY"))

    num_matches = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    for n in num_matches[:5]:
        entities.append(NamedEntityItem(text=n, label="CARDINAL"))

    return entities


def find_indicator_matches(text: str, patterns: List[str]) -> List[str]:
    """Find all matching substrings in text for a given list of regex patterns."""
    matched = []
    text_lower = text.lower()
    for pat in patterns:
        for match in re.finditer(pat, text_lower):
            span_text = text[match.start() : match.end()]
            if span_text not in matched:
                matched.append(span_text)
    return matched


def extract_nlp_features(text: str) -> NLPFeatures:
    """
    Complete NLP feature extraction pipeline combining statistical properties,
    sentiment, named entities, and categorized linguistic dark-pattern indicators.
    """
    prep = preprocess_text(text)
    original = prep["original_text"]
    tokens = prep["tokens"]
    sentences = prep["sentences"]

    char_count = len(original)
    word_count = len(tokens)
    sentence_count = max(1, len(sentences))
    avg_sentence_len = round(word_count / sentence_count, 2)

    # Punctuation and emphasis metrics
    number_count = len(re.findall(r"\b\d+\b", original))
    exclamation_count = original.count("!")
    question_count = original.count("?")

    # Uppercase analysis
    alpha_chars = [c for c in original if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = round(len(upper_chars) / max(1, len(alpha_chars)), 3)

    all_caps_words = [w for w in tokens if len(w) >= 2 and w.isupper() and w.isalpha()]
    uppercase_word_count = len(all_caps_words)

    # Sentiment analysis
    sentiment_score, sentiment_label = analyze_sentiment(original)

    # Named Entity Recognition
    named_entities = extract_named_entities(original)

    # Linguistic Indicators
    scarcity_matches = find_indicator_matches(original, INDICATOR_PATTERNS["scarcity"])
    urgency_matches = find_indicator_matches(original, INDICATOR_PATTERNS["urgency"])
    fomo_matches = find_indicator_matches(original, INDICATOR_PATTERNS["fomo"])
    social_matches = find_indicator_matches(original, INDICATOR_PATTERNS["social_proof"])
    shaming_matches = find_indicator_matches(original, INDICATOR_PATTERNS["confirmshaming"])
    continuity_matches = find_indicator_matches(original, INDICATOR_PATTERNS["forced_continuity"])

    total_indicators = (
        len(scarcity_matches)
        + len(urgency_matches)
        + len(fomo_matches)
        + len(social_matches)
        + len(shaming_matches)
        + len(continuity_matches)
    )

    # Deterministic NLP heuristic score calculation (0 - 100)
    # Combines indicators, punctuation emphasis, and urgency signals
    score = 0.0
    score += min(60.0, total_indicators * 18.0)
    score += min(15.0, exclamation_count * 5.0)
    score += min(15.0, uppercase_word_count * 5.0)
    if uppercase_ratio > 0.3:
        score += 10.0

    nlp_score = round(max(0.0, min(100.0, score)), 2)

    return NLPFeatures(
        character_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_length=avg_sentence_len,
        number_count=number_count,
        exclamation_count=exclamation_count,
        question_count=question_count,
        uppercase_word_count=uppercase_word_count,
        uppercase_ratio=uppercase_ratio,
        sentiment_score=sentiment_score,
        sentiment_label=sentiment_label,
        named_entities=named_entities,
        scarcity_indicators=scarcity_matches,
        urgency_indicators=urgency_matches,
        fomo_indicators=fomo_matches,
        social_proof_indicators=social_matches,
        confirmshaming_indicators=shaming_matches,
        forced_continuity_indicators=continuity_matches,
        total_indicator_count=total_indicators,
        nlp_heuristic_score=nlp_score,
    )
