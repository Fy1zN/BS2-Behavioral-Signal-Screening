"""
Utility functions for configuration loading, environment handling, and JSON parsing.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
from dotenv import load_dotenv

# Root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file automatically if present
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


def get_config_path() -> Path:
    """Return the absolute path to config.yaml."""
    return PROJECT_ROOT / "config" / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from config/config.yaml."""
    config_file = get_config_path()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration file format: root must be a mapping.")

    return config


def get_gemini_api_key() -> str:
    """
    Retrieve Gemini API key from environment variables.
    Raises ValueError if not configured.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to your .env file."
        )
    return api_key


def resolve_gemini_model(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Resolve the Gemini model using explicit priority:
    1. GEMINI_MODEL environment variable (if non-empty)
    2. config.yaml llm.model (if non-empty)
    3. Clear configuration error (no silent guessing)
    """
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model

    if config is None:
        config = load_config()

    cfg_model = config.get("llm", {}).get("model", "")
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()

    raise ValueError(
        "No Gemini model configured. Specify GEMINI_MODEL in your .env file or 'model' under 'llm' in config/config.yaml."
    )


def extract_json_from_text(raw_text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON object from raw LLM output text.
    Handles markdown fences (```json ... ```) or embedded JSON objects.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from LLM.")

    text = raw_text.strip()

    # Case 1: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Case 2: Markdown code fence extraction ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_matches = re.findall(fence_pattern, text, re.IGNORECASE)
    for match in fence_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Case 3: Find outermost curly braces { ... }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to extract valid JSON from LLM response text.")
