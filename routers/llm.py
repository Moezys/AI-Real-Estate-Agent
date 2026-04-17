"""
Gemini API integration for the two-stage LLM prompt chain.

Stage 1: extract_features() — parse user description into typed feature values
Stage 2: interpret_prediction() — explain the ML prediction in context
"""

import json
import logging
import os
import re
import time

from google import genai
from google.genai import types

from .config import GEMINI_MODEL
from .prompts import get_extraction_prompt, format_interpretation_prompt
from .schemas import ExtractedFeatures, ConversationMessage

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

def _call_with_retry(func, *args, **kwargs):
    """Call *func* with automatic retry on 429 RESOURCE_EXHAUSTED errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (attempt %d/%d), retrying in %ds…", attempt + 1, MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise
    # Final attempt — let it raise
    return func(*args, **kwargs)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response text, stripping markdown fences if present."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Stage 1 — Feature Extraction
# ---------------------------------------------------------------------------

def extract_features(
    messages: list[ConversationMessage],
    prompt_version: str = "v2",
) -> ExtractedFeatures:
    """
    Send conversation history to Gemini with the extraction system prompt.
    Returns ExtractedFeatures with completeness metadata.
    """
    client = _get_client()
    system_prompt = get_extraction_prompt(prompt_version)

    # Build contents list for the new SDK
    contents = []
    for msg in messages:
        role = "user" if msg.role == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

    try:
        response = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
        )
        raw_text = response.text
        logger.info("Stage 1 raw response: %s", raw_text[:500])

        parsed = _parse_json_response(raw_text)
        features = ExtractedFeatures(**parsed)
        features.compute_completeness()
        return features

    except json.JSONDecodeError as e:
        logger.error("Stage 1 JSON parse error: %s | raw: %s", e, raw_text[:300])
        fallback = ExtractedFeatures()
        fallback.compute_completeness()
        fallback.follow_up_message = (
            "I had a little trouble understanding that. "
            "Could you describe the property again? "
            "For example: '3-bedroom, 2-bath house in NAmes, built in 1990, about 1,500 sqft.'"
        )
        return fallback

    except Exception as e:
        logger.error("Stage 1 LLM error: %s", e)
        fallback = ExtractedFeatures()
        fallback.compute_completeness()
        fallback.follow_up_message = (
            "I'm having trouble connecting to our AI service right now. "
            "Please try again in a moment."
        )
        return fallback


# ---------------------------------------------------------------------------
# Stage 2 — Prediction Interpretation
# ---------------------------------------------------------------------------

def interpret_prediction(
    features: dict,
    predicted_price: float,
    training_stats: dict,
) -> str:
    """
    Feed extracted features, prediction, and training stats to Gemini.
    Returns a natural-language interpretation string.
    """
    client = _get_client()
    prompt = format_interpretation_prompt(features, predicted_price, training_stats)

    try:
        response = _call_with_retry(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a real estate analyst. Respond with plain text only.",
            ),
        )
        return response.text.strip()

    except Exception as e:
        logger.error("Stage 2 LLM error: %s", e)
        stats = training_stats["sale_price_stats"]
        median = stats["median"]
        diff_pct = ((predicted_price - median) / median) * 100
        direction = "above" if diff_pct > 0 else "below"
        return (
            f"The predicted price is ${predicted_price:,.0f}, "
            f"which is {abs(diff_pct):.1f}% {direction} the Ames median of ${median:,.0f}. "
            f"(Detailed interpretation unavailable — AI service error.)"
        )
