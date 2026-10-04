# BS² — Behavioral Signal Screening
### Explainable NLP and LLM-Based Detection of Manipulative Interface Language

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Google Gemini API](https://img.shields.io/badge/LLM-Google%20Gemini-orange.svg)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/NLP-spaCy-09A3D5.svg)](https://spacy.io/)
[![Pydantic v2](https://img.shields.io/badge/Validation-Pydantic%20v2-E92063.svg)](https://docs.pydantic.dev/)
[![Tests Passing](https://img.shields.io/badge/tests-37%20passed-brightgreen.svg)]()

---

## 1. Overview

**BS² (Behavioral Signal Screening)** is an academic Natural Language Processing (NLP) system designed to detect, screen, and explain deceptive interface language (commonly known as *dark patterns*) in web, mobile, e-commerce, and subscription flows.

Modern digital interfaces frequently employ subtle linguistic coercion—such as fabricated scarcity countdowns, guilt-tripping opt-out buttons, and obscured recurring fees—to subvert user autonomy. BS² addresses this problem using a defensible, multi-layered architecture:

1. **Deterministic NLP**: Extracting quantitative, reproducible linguistic signals (statistical text metrics, Named Entity Recognition via **spaCy**, sentiment polarity via **NLTK VADER**, and categorized lexical regex indicators).
2. **Contextual Semantic Reasoning**: Leveraging the **Google Gemini API** (using the official modern `google-genai` SDK) to perform multi-label classification, identify underlying psychological biases, extract textual evidence, and suggest transparent copywriting alternatives.
3. **Hybrid Explainable Scoring**: Fusing empirical NLP signals (30%) with Gemini semantic reasoning (70%) into a validated manipulation score (0–100) mapped to discrete severity tiers.

---

## 2. Problem Statement

Dark patterns are design and copywriting choices in user interfaces created to nudge or manipulate users into choices they might not otherwise make. While visual dark patterns (e.g., hidden checkboxes) have been studied, **textual manipulation** represents an equally pervasive threat that is harder to regulate algorithmically:
- **Scarcity & Urgency**: Fabricating artificial shortages ("*Only 2 rooms left!*") or false countdown deadlines to trigger panic.
- **Confirmshaming**: Emotionally manipulating users into opt-in decisions by making refusal feel shameful ("*No thanks, I hate saving money*").
- **Forced Continuity**: Concealing auto-renewal commitments beneath trial signups.

Traditional keyword blocklists suffer from high false-positive rates on benign factual text (e.g., "*Store closes at 9 PM*"), while standalone LLM prompts often hallucinate evidence or lack reproducibility. BS² bridges this divide through an explainable, hybrid NLP + LLM architecture.

---

## 3. Objectives

- **Empirical Signal Extraction**: Quantify measurable surface-level linguistic features before invoking neural reasoning.
- **Semantic Discrimination**: Distinguish genuine commercial communication (e.g., neutral order totals) from coercive manipulation.
- **Zero Hallucination Grounding**: Enforce that all detected evidence substrings strictly exist in the source input.
- **Explainable Diagnostics**: Deliver transparent, auditable rationales and psychological mechanisms exploited (e.g., Loss Aversion, FOMO, Sunk Cost).
- **Actionable Remediation**: Provide ethical, transparent copywriting alternatives for detected manipulative phrases.

---

## 4. NLP Techniques

BS² employs classical and statistical NLP techniques to build a deterministic feature representation:
- **Punctuation & Stylometric Analysis**: Measures exclamation density, question mark triggers, and uppercase token ratios (common indicators of urgency and emotional pressure).
- **Statistical Text Metrics**: Character counts, token counts, sentence segmentation boundaries, and average sentence lengths.
- **Lexical Dark Pattern Scanners**: Multi-vector regular expression heuristics scanning for scarcity cues, deadline pressure, FOMO triggers, social proof claims, confirmshaming grammar, and renewal traps.
- **Deterministic Heuristic Scoring**: A normalized 0–100 heuristic score based on indicator frequencies and typographic emphasis.

---

## 5. Pretrained Models

To satisfy academic project constraints without expensive and ungrounded model training from scratch, BS² integrates industry-standard pretrained NLP components:
- **spaCy (`en_core_web_sm`)**: Used for sentence segmentation, tokenization, and Named Entity Recognition (extracting `MONEY`, `DATE`, `CARDINAL`, `ORG`, `PERSON`, `GPE`, `PRODUCT`).
- **NLTK VADER (`SentimentIntensityAnalyzer`)**: A rule-based, pretrained sentiment analysis engine specifically calibrated for micro-texts, returning compound polarity and valence labels.

---

## 6. Google Gemini API Integration

The core semantic reasoning engine is powered strictly by **Google Gemini** using the official, current Python SDK (`google-genai`).

### Key Implementation Details:
- **SDK**: `google-genai>=1.0.0` (`from google import genai`).
- **Authentication**: Strict environment variable injection via `GEMINI_API_KEY`. No keys are ever hardcoded.
- **Configurability**: Model selection is dynamically resolved from `GEMINI_MODEL` environment variable, falling back to `config/config.yaml` (`gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-3.8-flash`, etc.).
- **Synchronous Execution**: Streamlined, synchronous single-call design avoiding unneeded background worker complexity.
- **Structured JSON Schema**: Prompts enforce raw JSON output matching our Pydantic schema.
- **Controlled Error Recovery**: If an initial response is malformed, a single recovery attempt is executed with feedback before triggering a graceful error state.
- **Comprehensive Failure Hierarchy**: Custom exceptions handling missing credentials (`GeminiConfigError`), invalid keys (`GeminiAuthError`), quota exhaustion (`GeminiRateLimitError`), timeouts (`GeminiTimeoutError`), and schema deviations (`GeminiResponseValidationError`).

---

## 7. Prompt Engineering

The complete prompt is decoupled from application logic and located at `prompts/bs2_analysis.txt`.

### Prompt Design Highlights:
- **Role Specification**: Defines BS² as an expert explainable NLP system for deceptive pattern detection.
- **Category Taxonomies**: Explicit definitions of all nine target categories.
- **Multimodal Inputs**: Accepts both original interface text and structured NLP summary metrics to ground reasoning in empirical features.
- **Strict Anti-Hallucination Guidelines**: Prohibits fabricating quotes not present in the input text.
- **Persuasion vs. Manipulation Boundary**: Explicitly instructs the model to classify standard, factual e-commerce statements as `NONE`.
- **Target Schema Enforcement**: Specifies required JSON keys: `is_dark_pattern`, `primary_category`, `secondary_categories`, `manipulation_score`, `severity`, `confidence`, `detected_phrases`, `psychological_mechanisms`, `explanation`, and `recommendation`.

---

## 8. Dark Pattern Categories

BS² categorizes interface language across 9 explicit taxonomies:

| Category | Description | Example |
| :--- | :--- | :--- |
| **SCARCITY** | Fabricating or exaggerating limited stock to induce panic buying. | *"Only 2 rooms left at this price!"* |
| **URGENCY** | Imposing artificial deadlines or countdowns without genuine cause. | *"Flash Sale ends in 04:59 minutes! Act now!"* |
| **CONFIRMSHAMING** | Emotionally manipulating opt-out buttons to induce guilt. | *"No thanks, I hate saving money."* |
| **HIDDEN_COST** | Concealing mandatory surcharges until the final checkout step. | *"Mandatory resort and service fees added at checkout."* |
| **FORCED_CONTINUITY** | Auto-enrolling users into recurring billing after a trial. | *"Free trial automatically renews at ₹1,499/mo."* |
| **MISDIRECTION** | Using confusing double negatives or asymmetric visual cues. | *"Uncheck if you do not wish to not receive ads."* |
| **SOCIAL_PROOF_MANIPULATION**| Weaponizing live viewer counters to trigger herd behavior. | *"47 people are viewing this product right now!"* |
| **OBSTRUCTION** | Creating severe asymmetric friction to cancel or delete an account. | *"Call our hotline during business hours to cancel."* |
| **NONE** | Transparent, neutral, factual, or benign commercial text. | *"Your order total is ₹999 including taxes."* |

---

## 9. Architecture

```text
                USER INTERFACE TEXT
                         │
                         ▼
                 INPUT VALIDATION
           (Length, empty check, sanitization)
                         │
                         ▼
                 TRADITIONAL NLP
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Tokens       spaCy NER      NLTK VADER
          │              │              │
          └──────────────┼──────────────┘
                         ▼
             LINGUISTIC INDICATORS
     (Scarcity, Urgency, FOMO, Social Proof, etc.)
                         │
                         ▼
             TOKEN-EFFICIENT PAYLOAD
                         │
                         ▼
             ┌───────────────────────┐
             │   GOOGLE GEMINI API   │
             │     (google-genai)    │
             └───────────┬───────────┘
                         ▼
             CONTEXTUAL REASONING
                         │
                         ▼
              STRUCTURED JSON OUTPUT
                         │
                         ▼
               PYDANTIC VALIDATION
             (LLMAnalysis Data Model)
                         │
                         ▼
                 HYBRID SCORING
     Final = 0.70 × Gemini Score + 0.30 × NLP Score
                         │
                         ▼
               EXPLAINABLE REPORT
                         │
                         ▼
                    STREAMLIT UI
```

---

## 10. Scoring Methodology

The final manipulation score ($S_{\text{final}} \in [0, 100]$) is computed through a convex combination of deterministic NLP features and Gemini contextual assessment:

$$S_{\text{final}} = w_{\text{llm}} \cdot S_{\text{llm}} + w_{\text{nlp}} \cdot S_{\text{nlp}}$$

Configured in `config/config.yaml`:
- $w_{\text{llm}} = 0.70$
- $w_{\text{nlp}} = 0.30$

### Severity Tiers:
- **0 – 19: NONE** (Neutral, factual communication)
- **20 – 39: LOW** (Mild marketing persuasion, non-coercive)
- **40 – 59: MEDIUM** (Borderline behavioral nudging)
- **60 – 79: HIGH** (Clear manipulative dark pattern)
- **80 – 100: CRITICAL** (Aggressive, multi-layered deceptive coercion)

---

## 11. Error Handling & Resilience

| Scenario | Handling Strategy |
| :--- | :--- |
| **Missing API Key** | Pre-flight check detects absence and halts execution without network requests, displaying instructions. |
| **Invalid API Key** | `GeminiAuthError` caught with instructions to check credentials in `.env`. |
| **Rate Limiting (429)** | `GeminiRateLimitError` gracefully notifies the user without entering retry storms. |
| **Model Unavailable (404)** | `GeminiModelNotFoundError` indicates the configured model name must be corrected. |
| **Request Timeout** | Synchronous call guarded by configurable 30s timeout (`GeminiTimeoutError`). |
| **Malformed JSON** | A single controlled recovery prompt is dispatched; if still invalid, a validation error is reported. |
| **Oversized Input (>6000 chars)** | Input validation halts processing before token consumption. |

---

## 12. Project Structure

```text
BS2/
├── app.py                      # Streamlit interactive application
├── requirements.txt            # Locked project dependencies
├── README.md                   # Comprehensive documentation & rubric mapping
├── .gitignore                  # Git privacy & secret protection
├── .env.example                # Environment variable template
├── PROJECT_STATE.md            # Continual state tracking and verification log
│
├── config/
│   └── config.yaml             # Model, weight, threshold, and category configuration
│
├── prompts/
│   └── bs2_analysis.txt        # Decoupled prompt engineering template
│
├── src/
│   ├── __init__.py             # Package initializer
│   ├── models.py               # Pydantic v2 schemas (NLPFeatures, LLMAnalysis, FinalAnalysis)
│   ├── utils.py                # Configuration loader, model resolver, and JSON extractor
│   ├── preprocessing.py        # Input validation, sentence splitting, tokenization
│   ├── nlp_features.py         # Statistical features, spaCy NER, VADER sentiment, indicators
│   ├── llm_analyzer.py         # Google Gemini API client, recovery, and exception hierarchy
│   └── scoring.py              # Hybrid scoring, severity mapping, FinalAnalysis assembler
│
├── data/
│   └── test_cases.json         # 22 curated benchmark scenarios across all categories
│
└── tests/
    ├── test_preprocessing.py   # Tests for input validation, segmentation, and normalization
    ├── test_features.py        # Tests for statistical metrics, NER, sentiment, and indicators
    ├── test_models.py          # Tests for strict Pydantic bounds and type enforcement
    ├── test_scoring.py         # Tests for hybrid weights, severity boundaries, and clamping
    └── test_llm_analyzer.py    # Mock tests for Gemini API, single-recovery, and error handling
```

---

## 13. Installation

### 1. Clone or Navigate to Repository
```bash
cd BS2
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Pretrained NLP Models
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## 14. Environment Variables

Create a `.env` file in the project root based on `.env.example`:

```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

> **Security Note**: Never commit `.env` or expose API keys in code or documentation.

---

## 15. Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 16. Testing

The project includes a comprehensive unit test suite covering input validation, NLP feature extraction, Pydantic constraints, scoring boundary math, and Gemini API error simulation via mocks.

Run the test suite:

```bash
pytest tests -v
```

All 37 unit tests run completely offline without requiring an active API key or consuming API quotas.

---

## 17. Example Results

### Example 1: High Scarcity & Urgency
- **Input Text**: `"Only 2 rooms left at this price! Book now before someone else takes your room."`
- **Manipulation Score**: `75.3 / 100` (High)
- **Primary Pattern**: `SCARCITY`
- **Secondary Pattern**: `URGENCY`
- **Detected Evidence**: `"Only 2 rooms left"`, `"Book now before someone else takes your room"`
- **Psychological Mechanisms**: Loss Aversion, Fear of Missing Out (FOMO), Time Pressure Panic
- **Recommendation**: *"Present room availability neutrally without artificial countdowns or social comparison triggers."*

### Example 2: Non-Manipulative Order Receipt
- **Input Text**: `"Your order total is ₹999 including all applicable taxes and standard delivery."`
- **Manipulation Score**: `7.0 / 100` (None)
- **Primary Pattern**: `NONE`
- **Classification**: `NON-MANIPULATIVE`
- **Psychological Mechanisms**: None detected
- **Recommendation**: *"Maintain transparent, factual receipt formatting."*

---

## 18. Limitations

1. **Textual Focus**: BS² analyzes textual language; visual dark patterns (e.g., low-contrast gray buttons or tiny font sizes) require computer vision analysis.
2. **Dynamic Context**: Static text snippets do not capture multi-page funnel tricks (e.g., hidden checkboxes across multiple steps).
3. **Language Scope**: Currently optimized for English interface copy.

---

## 19. Future Scope

- **Visual Multimodal Analysis**: Utilizing Gemini's vision capabilities to evaluate UI layouts, button colors, and visual hierarchy.
- **Browser Extension**: Packaging BS² into a lightweight Chrome extension that highlights manipulative text on e-commerce sites in real time.
- **Multilingual Support**: Extending linguistic indicator sets to regional languages.

---

## 20. Rubric Mapping for Academic Evaluation

| Evaluation Criteria | Project Implementation & Proof Points |
| :--- | :--- |
| **1. Code Quality & Architecture** | Modular design across `src/`, separated concerns, Pydantic v2 data models, explicit exception hierarchies, 37 passing unit tests, typed signatures, and full docstrings. |
| **2. Google Gemini API Integration** | Official modern `google-genai` SDK, synchronous requests, externalized configuration (`config.yaml`), strict environment-variable credential loading, and single-attempt controlled recovery. |
| **3. Prompt Effectiveness & Efficiency** | Decoupled prompt deliverable (`prompts/bs2_analysis.txt`), grounded anti-hallucination rules, token-efficient payload summarization, and strict JSON output schema. |
| **4. Overall Project Quality** | Dual-layer hybrid scoring combining spaCy/VADER NLP (30%) with Gemini semantic reasoning (70%), responsive Streamlit UI with evaluators' debug panel, benchmark dataset (`data/test_cases.json`), and comprehensive documentation. |
