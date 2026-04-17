"""
Chat router — the core prompt chain endpoint.

Flow:
  User message → Security check → Stage 1 LLM (extract features)
    → if incomplete: return follow-up question (conversational loop)
    → if complete:   ML predict → Stage 2 LLM (interpret) → return prediction
"""

import logging

from fastapi import APIRouter

from .schemas import (
    ChatRequest,
    ExtractedFeatures,
    FollowUpResponse,
    PredictionResponse,
)
from .security import validate_user_input
from .llm import extract_features, interpret_prediction
from .ml_model import predict_price, get_training_stats
from .config import DEFAULT_PROMPT_VERSION

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat")
def chat(request: ChatRequest) -> PredictionResponse | FollowUpResponse:
    """
    Main prompt-chain endpoint.

    Accepts conversation history, returns either:
    - FollowUpResponse (features incomplete → assistant asks for more)
    - PredictionResponse (features complete → price + interpretation)
    """
    if not request.messages:
        return FollowUpResponse(
            extracted_features=ExtractedFeatures(),
            assistant_message="Hi! I'm your AI real estate agent for Ames, Iowa. "
            "Describe a property and I'll predict its price. For example: "
            "'3-bedroom house with 2 baths in NAmes, about 1,500 sqft, built in 1990.'",
        )

    # --- Security: validate the latest user message ---
    last_user_msg = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg
            break

    if last_user_msg:
        clean_text, is_safe = validate_user_input(last_user_msg.content)
        if not is_safe:
            logger.warning("Potential injection detected: %s", clean_text[:100])
            return FollowUpResponse(
                extracted_features=ExtractedFeatures(),
                assistant_message="I'm here to help with property price predictions in Ames, Iowa. "
                "Could you describe the property you'd like me to evaluate? "
                "For example: number of bedrooms, bathrooms, square footage, neighborhood, etc.",
            )

    # --- Stage 1: Extract features via LLM ---
    features = extract_features(request.messages, prompt_version=DEFAULT_PROMPT_VERSION)

    # --- If incomplete: return follow-up (conversational loop) ---
    if features.confidence != "complete":
        follow_up = features.follow_up_message or (
            f"Thanks! I've got {len(features.extracted_features)} of 12 details so far. "
            f"I still need: {', '.join(features.missing_features[:4])}. "
            "Could you fill in a few more?"
        )
        return FollowUpResponse(
            extracted_features=features,
            assistant_message=follow_up,
        )

    # --- ML Prediction ---
    predicted_price = predict_price(features)

    # --- Stage 2: Interpret prediction via LLM ---
    # Build features dict (only the 12 model features, no metadata)
    features_dict = {
        name: getattr(features, name)
        for name in features.all_feature_names
        if getattr(features, name) is not None
    }

    training_stats = get_training_stats()
    interpretation = interpret_prediction(features_dict, predicted_price, training_stats)

    # Build market context
    stats = training_stats["sale_price_stats"]
    neighborhood = features.Neighborhood
    market_context = {
        "ames_median_price": stats["median"],
        "ames_mean_price": stats["mean"],
        "price_vs_median_pct": round(
            ((predicted_price - stats["median"]) / stats["median"]) * 100, 1
        ),
    }
    if neighborhood and neighborhood in training_stats.get("neighborhood_stats", {}):
        nb = training_stats["neighborhood_stats"][neighborhood]
        market_context["neighborhood_median"] = nb["median"]
        market_context["neighborhood_avg"] = nb["mean"]
        market_context["neighborhood_sales_count"] = int(nb["count"])

    return PredictionResponse(
        extracted_features=features,
        predicted_price=round(predicted_price, 2),
        interpretation=interpretation,
        market_context=market_context,
    )
