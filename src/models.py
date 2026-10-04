"""
Data models for BS² (Behavioral Signal Screening).
Provides strict Pydantic validation for NLP features, Gemini LLM output, and final aggregated analysis.
"""

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

DarkPatternCategory = Literal[
    "SCARCITY",
    "URGENCY",
    "CONFIRMSHAMING",
    "HIDDEN_COST",
    "FORCED_CONTINUITY",
    "MISDIRECTION",
    "SOCIAL_PROOF_MANIPULATION",
    "OBSTRUCTION",
    "NONE"
]

SeverityLevel = Literal[
    "NONE",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL"
]

SentimentLabel = Literal[
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE"
]


class NamedEntityItem(BaseModel):
    """Structured representation of a recognized named entity."""
    text: str
    label: str


class NLPFeatures(BaseModel):
    """Measurable linguistic indicators and statistical features extracted via spaCy and NLTK."""
    character_count: int = Field(ge=0, description="Total characters in input")
    word_count: int = Field(ge=0, description="Total words in input")
    sentence_count: int = Field(ge=0, description="Total sentences segmented")
    avg_sentence_length: float = Field(ge=0.0, description="Average words per sentence")

    number_count: int = Field(ge=0, description="Count of numerical digits and quantitative tokens")
    exclamation_count: int = Field(ge=0, description="Count of exclamation marks")
    question_count: int = Field(ge=0, description="Count of question marks")
    uppercase_word_count: int = Field(ge=0, description="Words in ALL-CAPS for emphasis")
    uppercase_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of uppercase characters to alphabetic characters")

    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Sentiment compound polarity (-1.0 to +1.0)")
    sentiment_label: SentimentLabel = Field(default="NEUTRAL", description="Categorical sentiment label")

    named_entities: List[NamedEntityItem] = Field(default_factory=list, description="Entities detected by spaCy")

    scarcity_indicators: List[str] = Field(default_factory=list, description="Matched scarcity keywords")
    urgency_indicators: List[str] = Field(default_factory=list, description="Matched urgency keywords")
    fomo_indicators: List[str] = Field(default_factory=list, description="Matched fear-of-missing-out phrases")
    social_proof_indicators: List[str] = Field(default_factory=list, description="Matched social proof patterns")
    confirmshaming_indicators: List[str] = Field(default_factory=list, description="Matched confirmshaming patterns")
    forced_continuity_indicators: List[str] = Field(default_factory=list, description="Matched forced continuity patterns")

    total_indicator_count: int = Field(ge=0, description="Sum of all matched linguistic dark pattern indicators")
    nlp_heuristic_score: float = Field(ge=0.0, le=100.0, description="Deterministic NLP score normalized to 0-100")


class LLMAnalysis(BaseModel):
    """Structured output returned and validated from Google Gemini API."""
    is_dark_pattern: bool = Field(description="Whether the text contains a manipulative dark pattern")
    primary_category: DarkPatternCategory = Field(description="Primary dark pattern category or NONE")
    secondary_categories: List[DarkPatternCategory] = Field(default_factory=list, description="Secondary categories identified")
    manipulation_score: float = Field(ge=0.0, le=100.0, description="Manipulation severity score from 0 to 100")
    severity: SeverityLevel = Field(description="Severity tier: NONE, LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in classification from 0.0 to 1.0")
    detected_phrases: List[str] = Field(default_factory=list, description="Exact substrings extracted as evidence")
    psychological_mechanisms: List[str] = Field(default_factory=list, description="Psychological biases exploited")
    explanation: str = Field(min_length=5, description="Explainable rationale of the semantic classification")
    recommendation: str = Field(min_length=5, description="Actionable alternative respecting user autonomy")

    @field_validator("manipulation_score", mode="before")
    @classmethod
    def validate_score(cls, v):
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid manipulation_score: {v}")
        if val < 0.0 or val > 100.0:
            raise ValueError(f"manipulation_score must be between 0 and 100, received: {val}")
        return round(val, 2)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, v):
        try:
            val = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid confidence: {v}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, received: {val}")
        return round(val, 4)

    @field_validator("primary_category", mode="before")
    @classmethod
    def normalize_primary_category(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().upper()
            return v_clean
        return v

    @field_validator("secondary_categories", mode="before")
    @classmethod
    def normalize_secondary_categories(cls, v):
        if not v:
            return []
        cleaned = []
        for cat in v:
            if isinstance(cat, str):
                cleaned.append(cat.strip().upper())
            else:
                cleaned.append(cat)
        return cleaned

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().upper()
            return v_clean
        return v


class FinalAnalysis(BaseModel):
    """Aggregated explainable analysis combining NLP features and Google Gemini semantic reasoning."""
    input_text: str
    nlp_features: NLPFeatures
    llm_analysis: LLMAnalysis

    nlp_score: float = Field(ge=0.0, le=100.0)
    llm_score: float = Field(ge=0.0, le=100.0)
    final_score: float = Field(ge=0.0, le=100.0)
    severity: SeverityLevel
    is_manipulative: bool

    primary_category: DarkPatternCategory
    secondary_categories: List[DarkPatternCategory] = Field(default_factory=list)
    detected_evidence: List[str] = Field(default_factory=list)
    psychological_mechanisms: List[str] = Field(default_factory=list)

    explanation: str
    recommendation: str

    weights_used: Dict[str, float] = Field(default_factory=lambda: {"llm_weight": 0.70, "nlp_weight": 0.30})
    model_used: str = "gemini-2.5-flash"
    execution_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
