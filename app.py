"""
BS² — Behavioral Signal Screening
Explainable NLP and LLM-Based Detection of Manipulative Interface Language
Streamlit Web Application
"""

import json
import os
import streamlit as st

from src.llm_analyzer import (
    GeminiAnalyzer,
    GeminiAnalyzerError,
    GeminiAuthError,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiRateLimitError,
    GeminiResponseValidationError,
    GeminiTimeoutError,
)
from src.nlp_features import extract_nlp_features
from src.preprocessing import validate_input_text
from src.scoring import build_final_analysis
from src.utils import get_gemini_api_key, load_config, resolve_gemini_model

# Page Configuration
st.set_page_config(
    page_title="BS² — Behavioral Signal Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic Polish
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .tagline {
        font-size: 0.95rem;
        color: #64748B;
        font-style: italic;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-low {
        background-color: #E0E7FF;
        color: #3730A3;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-none {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">BS²</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Behavioral Signal Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Explainable NLP and LLM-Based Detection of Manipulative Interface Language</div>',
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.header(" Configuration & Diagnostics")

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        config = {}
        st.error(f"Config load error: {e}")

    default_model = config.get("llm", {}).get("model", "gemini-2.5-flash")

    # API Key Handling
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=env_api_key,
        type="password",
        help="Reads from .env by default. Enter your API key from Google AI Studio if not set in .env.",
    )

    # Model Selection
    model_options = [
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-3.8-flash",
        "gemini-1.5-pro",
    ]
    model_choice = st.selectbox(
        "Gemini Model",
        options=model_options,
        index=0 if default_model not in model_options else model_options.index(default_model),
        help="Configured in config.yaml or overridden here.",
    )

    st.markdown("---")
    st.subheader("Architecture Overview")
    st.markdown(
        """
        - **Deterministic NLP**: spaCy NER, tokenization, statistical metrics, NLTK VADER sentiment
        - **Linguistic Scanners**: Heuristic indicators across 6 dark-pattern vectors
        - **Google Gemini API**: Contextual reasoning & multi-label classification
        - **Pydantic Validation**: Strict schema enforcement & error recovery
        - **Hybrid Scoring**: `0.70 × Gemini + 0.30 × NLP`
        """
    )
    st.caption("College NLP Academic Project")

# Pre-packaged Realistic Examples
EXAMPLES = {
    "Select an example...": "",
    "Scarcity Pattern": "Only 2 rooms left at this price! Book now before someone else takes your room.",
    "Urgency Pattern": "Flash Sale ends in 04:59 minutes! Complete your checkout immediately or lose your 40% discount forever.",
    "Confirmshaming Pattern": "No thanks, I don't want to save money and prefer paying full price.",
    "Forced Continuity": "Start your free 7-day trial now. Your credit card will be automatically charged ₹1,499/month unless cancelled in account settings prior to renewal.",
    "Social Proof Manipulation": "47 people are viewing this product right now! 12 customers booked in the last 15 minutes.",
    "Normal Factual Text": "Your order total is ₹999 including all applicable taxes and standard delivery.",
}

# Session State Initialization
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Example Selector
st.markdown("#####  Test with Benchmark Interface Scenarios")
cols = st.columns(len(EXAMPLES) - 1)
example_keys = list(EXAMPLES.keys())[1:]

for i, key in enumerate(example_keys):
    if cols[i].button(key, use_container_width=True):
        st.session_state.user_text = EXAMPLES[key]
        st.session_state.analysis_result = None
        st.rerun()

# Input Area
st.markdown("##### 📝 Input Interface Language")
input_text = st.text_area(
    label="Paste interface copy, e-commerce checkout text, cookie consent dialog, or subscription terms:",
    value=st.session_state.user_text,
    height=130,
    placeholder="e.g. Only 2 items left in stock! Hurry, sale ends in 5 minutes...",
)
st.session_state.user_text = input_text

col_btn1, col_btn2, _ = st.columns([1, 1, 4])
analyze_clicked = col_btn1.button(" Analyze Language", type="primary", use_container_width=True)
clear_clicked = col_btn2.button(" Clear", use_container_width=True)

if clear_clicked:
    st.session_state.user_text = ""
    st.session_state.analysis_result = None
    st.rerun()

# Execution Flow
if analyze_clicked:
    # Step 1: Input Validation
    max_len = int(config.get("analysis", {}).get("max_text_length", 6000))
    is_valid, validation_error = validate_input_text(input_text, max_length=max_len)

    if not is_valid:
        st.error(f" Validation Error: {validation_error}")
    elif not api_key_input:
        st.error(
            " Gemini API key is not configured. Add GEMINI_API_KEY to your .env file or enter it in the sidebar."
        )
    else:
        with st.spinner("Executing NLP Pipeline & Calling Google Gemini API..."):
            try:
                # Step 2: Traditional NLP Feature Extraction
                nlp_features = extract_nlp_features(input_text)

                # Step 3: Google Gemini API Analysis
                analyzer = GeminiAnalyzer(api_key=api_key_input, model=model_choice)
                llm_output = analyzer.analyze(input_text, nlp_features)

                # Step 4: Hybrid Scoring and Final Assembly
                final_result = build_final_analysis(
                    input_text=input_text,
                    nlp_features=nlp_features,
                    llm_analysis=llm_output,
                    model_name=model_choice,
                    config=config,
                )
                st.session_state.analysis_result = final_result

            except GeminiConfigError as e:
                st.error(f" Configuration Error: {e}")
            except GeminiAuthError as e:
                st.error(f" Authentication Error: {e}")
            except GeminiRateLimitError as e:
                st.warning(f" Rate Limit Warning: {e}")
            except GeminiModelNotFoundError as e:
                st.error(f" Model Error: {e}")
            except GeminiTimeoutError as e:
                st.error(f" Timeout: {e}")
            except GeminiResponseValidationError as e:
                st.error(f" Validation Failure: {e}")
            except GeminiAnalyzerError as e:
                st.error(f" LLM Error: {e}")
            except Exception as e:
                st.error(f" Unexpected Execution Error: {str(e)}")

# Display Results
result = st.session_state.analysis_result
if result:
    st.markdown("---")
    st.subheader(" Behavioral Analysis Report")

    # Score and Severity Top Badges
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)

    severity_badge_class = f"badge-{result.severity.lower()}"
    status_label = "MANIPULATIVE" if result.is_manipulative else "NON-MANIPULATIVE"
    status_color = "#DC2626" if result.is_manipulative else "#16A34A"

    with score_col1:
        st.metric(
            label="Manipulation Score",
            value=f"{result.final_score:.1f} / 100",
            delta=f"NLP: {result.nlp_score:.1f} | Gemini: {result.llm_score:.1f}",
        )

    with score_col2:
        st.markdown(
            f"""
            <div style="padding-top: 5px;">
                <span style="font-size: 0.85rem; color: #64748B; font-weight: 600;">CLASSIFICATION</span><br/>
                <span style="font-size: 1.4rem; font-weight: 800; color: {status_color};">{status_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col3:
        st.markdown(
            f"""
            <div style="padding-top: 5px;">
                <span style="font-size: 0.85rem; color: #64748B; font-weight: 600;">SEVERITY TIER</span><br/>
                <span class="{severity_badge_class}" style="font-size: 1.1rem; margin-top: 4px;">{result.severity}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with score_col4:
        st.metric(
            label="Model Confidence",
            value=f"{result.llm_analysis.confidence * 100:.1f}%",
            delta=f"Model: {result.model_used}",
        )

    # Categories
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        st.markdown(f"**Primary Pattern:** `{result.primary_category}`")
    with cat_col2:
        sec_text = ", ".join(result.secondary_categories) if result.secondary_categories else "None"
        st.markdown(f"**Secondary Pattern(s):** `{sec_text}`")

    # Psychological Mechanisms & Detected Evidence
    content_col1, content_col2 = st.columns(2)

    with content_col1:
        st.markdown("##### 🎯 Psychological Mechanisms Exploited")
        if result.psychological_mechanisms:
            for mech in result.psychological_mechanisms:
                st.markdown(f"- **{mech}**")
        else:
            st.info("No manipulative cognitive biases detected.")

    with content_col2:
        st.markdown("##### 🔎 Detected Textual Evidence")
        if result.detected_evidence:
            for ev in result.detected_evidence:
                st.markdown(f"- ❝ *{ev}* ❞")
        else:
            st.info("No manipulative phrases detected.")

    # Explanation and Recommendation
    st.markdown("#####  Semantic Explanation (Google Gemini)")
    st.info(result.explanation)

    st.markdown("#####  Transparent Recommendation")
    st.success(result.recommendation)

    # Measurable NLP Signals
    st.markdown("#####  Measurable NLP Signals (spaCy & NLTK)")
    nlp = result.nlp_features
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    stat_col1.metric("Word Count", nlp.word_count)
    stat_col2.metric("Sentences", nlp.sentence_count)
    stat_col3.metric("Exclamations", nlp.exclamation_count)
    stat_col4.metric("Uppercase Ratio", f"{nlp.uppercase_ratio * 100:.1f}%")
    stat_col5.metric("Sentiment", f"{nlp.sentiment_label} ({nlp.sentiment_score:+.2f})")

    if nlp.named_entities:
        ent_str = ", ".join([f"**{e.text}** ({e.label})" for e in nlp.named_entities])
        st.caption(f"**Recognized Entities:** {ent_str}")

    # Technical Analysis (Evaluator Section)
    with st.expander(" Technical Inspection & API Details (Evaluator View)"):
        st.markdown(
            f"""
            - **Google Gemini Provider**: Verified official `google-genai` integration
            - **Model Invoked**: `{result.model_used}`
            - **Scoring Breakdown**:
              - Deterministic NLP Heuristic Score: **{result.nlp_score:.2f} / 100** (Weight: {result.weights_used['nlp_weight'] * 100:.0f}%)
              - Contextual Gemini Semantic Score: **{result.llm_score:.2f} / 100** (Weight: {result.weights_used['llm_weight'] * 100:.0f}%)
              - Final Aggregate Manipulation Score: **{result.final_score:.2f} / 100**
            - **Execution Timestamp**: `{result.execution_timestamp}`
            """
        )
        st.markdown("**Validated Google Gemini Output Schema (Pydantic):**")
        st.json(result.llm_analysis.model_dump())
