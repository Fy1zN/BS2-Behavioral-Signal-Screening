"""
Google Gemini API integration module for BS².
Executes synchronous semantic analysis, structured JSON parsing,
Pydantic validation, and controlled recovery for dark-pattern classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.models import LLMAnalysis, NLPFeatures
from src.utils import (
    extract_json_from_text,
    get_gemini_api_key,
    load_config,
    resolve_gemini_model,
)

# Custom Exception Hierarchy for Explicit, Informative Failure Handling
class GeminiAnalyzerError(Exception):
    """Base exception for Gemini analyzer failures."""
    pass


class GeminiConfigError(GeminiAnalyzerError):
    """Raised when API key or model configuration is missing or malformed."""
    pass


class GeminiAuthError(GeminiAnalyzerError):
    """Raised when API key authentication fails."""
    pass


class GeminiRateLimitError(GeminiAnalyzerError):
    """Raised when API quota or rate limits are exceeded."""
    pass


class GeminiTimeoutError(GeminiAnalyzerError):
    """Raised when the Gemini API call exceeds the configured timeout."""
    pass


class GeminiModelNotFoundError(GeminiAnalyzerError):
    """Raised when the configured Gemini model is unavailable or unrecognized."""
    pass


class GeminiResponseValidationError(GeminiAnalyzerError):
    """Raised when Gemini response fails JSON parsing or Pydantic validation."""
    pass


def load_prompt_template() -> str:
    """Load the external prompt template from prompts/bs2_analysis.txt."""
    project_root = Path(__file__).resolve().parent.parent
    prompt_path = project_root / "prompts" / "bs2_analysis.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"External prompt file not found at: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def build_analysis_prompt(text: str, nlp_features: NLPFeatures) -> str:
    """
    Format the external prompt template with original text and token-efficient NLP features.
    """
    template = load_prompt_template()

    # Create compact, token-efficient representation of relevant NLP features
    nlp_summary = {
        "word_count": nlp_features.word_count,
        "sentence_count": nlp_features.sentence_count,
        "exclamation_count": nlp_features.exclamation_count,
        "uppercase_word_count": nlp_features.uppercase_word_count,
        "sentiment_polarity": nlp_features.sentiment_score,
        "sentiment_label": nlp_features.sentiment_label,
        "detected_entities": [
            f"{e.label}: {e.text}" for e in nlp_features.named_entities[:8]
        ],
        "linguistic_indicators": {
            "scarcity": nlp_features.scarcity_indicators,
            "urgency": nlp_features.urgency_indicators,
            "fomo": nlp_features.fomo_indicators,
            "social_proof": nlp_features.social_proof_indicators,
            "confirmshaming": nlp_features.confirmshaming_indicators,
            "forced_continuity": nlp_features.forced_continuity_indicators,
        },
        "nlp_heuristic_score": nlp_features.nlp_heuristic_score,
    }

    nlp_json_str = json.dumps(nlp_summary, indent=2)
    prompt = template.replace("{original_text}", text).replace(
        "{nlp_features_json}", nlp_json_str
    )
    return prompt


class GeminiAnalyzer:
    """
    Synchronous analyzer orchestrating Google Gemini API calls,
    structured output enforcement, and single-attempt recovery.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.config = load_config()

        try:
            self.api_key = api_key or get_gemini_api_key()
        except ValueError as e:
            raise GeminiConfigError(str(e))

        try:
            self.model = model or resolve_gemini_model(self.config)
        except ValueError as e:
            raise GeminiConfigError(str(e))

        self.temperature = float(self.config.get("llm", {}).get("temperature", 0.1))
        self.max_tokens = int(self.config.get("llm", {}).get("max_tokens", 1500))
        self.timeout_seconds = int(self.config.get("llm", {}).get("timeout_seconds", 30))

        # Lazy initialize client
        self._client = None

    def _get_client(self):
        """Initialize Google Gemini client using official google-genai SDK."""
        if self._client is not None:
            return self._client

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            return self._client
        except ImportError:
            # Check for legacy google.generativeai if google-genai is somehow absent
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self._client = legacy_genai
                return self._client
            except ImportError:
                raise GeminiConfigError(
                    "Google Gemini SDK is not installed. Please install 'google-genai' via pip."
                )

    def _call_gemini_api(self, prompt: str) -> str:
        """
        Execute synchronous request to Gemini API.
        Enforces timeout and translates API exceptions.
        """
        client = self._get_client()

        try:
            # Modern google-genai SDK
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                from google.genai import types

                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                )
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                if not response or not response.text:
                    raise GeminiResponseValidationError("Gemini API returned an empty response.")
                return response.text

            # Legacy fallback if applicable
            elif hasattr(client, "GenerativeModel"):
                model = client.GenerativeModel(
                    self.model,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                        "response_mime_type": "application/json",
                    },
                )
                response = model.generate_content(prompt)
                if not response or not response.text:
                    raise GeminiResponseValidationError("Gemini API returned an empty response.")
                return response.text

            else:
                raise GeminiConfigError("Unrecognized Gemini SDK client structure.")

        except Exception as e:
            err_msg = str(e).lower()
            if "401" in err_msg or "api_key_invalid" in err_msg or "unauthenticated" in err_msg or "invalid api key" in err_msg:
                raise GeminiAuthError(
                    "Invalid Google Gemini API key. Please check your GEMINI_API_KEY in .env."
                ) from e
            elif "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                raise GeminiRateLimitError(
                    "Gemini API rate limit or quota exceeded. Please wait before making further requests."
                ) from e
            elif "404" in err_msg or "not found" in err_msg or "model" in err_msg and "does not exist" in err_msg:
                raise GeminiModelNotFoundError(
                    f"Configured Gemini model '{self.model}' was not found or is unavailable. Please check GEMINI_MODEL in .env or config.yaml."
                ) from e
            elif "timeout" in err_msg or "timed out" in err_msg or "deadline" in err_msg:
                raise GeminiTimeoutError(
                    f"Gemini API call timed out after {self.timeout_seconds} seconds. Please try again."
                ) from e
            elif isinstance(e, (GeminiConfigError, GeminiAuthError, GeminiRateLimitError, GeminiModelNotFoundError, GeminiTimeoutError, GeminiResponseValidationError)):
                raise e
            else:
                raise GeminiAnalyzerError(f"Gemini API error occurred: {str(e)}") from e

    def analyze(self, text: str, nlp_features: NLPFeatures) -> LLMAnalysis:
        """
        Analyze interface text using Google Gemini with structured Pydantic validation
        and a single recovery attempt in case of malformed output.
        """
        # Step 1: Build formatted prompt
        prompt = build_analysis_prompt(text, nlp_features)

        # Step 2: First attempt
        raw_output = self._call_gemini_api(prompt)

        # Step 3: Parse and validate JSON
        try:
            data = extract_json_from_text(raw_output)
            return LLMAnalysis(**data)
        except Exception as first_err:
            # Step 4: Controlled single recovery attempt
            recovery_prompt = (
                f"{prompt}\n\n"
                f"ATTENTION: Your previous response failed validation with error: {str(first_err)}.\n"
                f"You MUST return ONLY valid raw JSON matching the required schema without any surrounding markdown or prose."
            )
            try:
                recovery_output = self._call_gemini_api(recovery_prompt)
                recovery_data = extract_json_from_text(recovery_output)
                return LLMAnalysis(**recovery_data)
            except Exception as final_err:
                raise GeminiResponseValidationError(
                    f"Failed to obtain valid LLM analysis after recovery attempt: {str(final_err)}"
                ) from final_err
