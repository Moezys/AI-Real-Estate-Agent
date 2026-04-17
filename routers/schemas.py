from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# --- Conversation ---

class ConversationMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ConversationMessage]


# --- Stage 1: Feature Extraction ---

class ExtractedFeatures(BaseModel):
    """Pydantic schema #1 — what Stage 1 LLM returns."""

    # 11 numeric features (all Optional — user may not mention them)
    GrLivArea: Optional[float] = Field(None, description="Above-grade living area sqft")
    TotalBsmtSF: Optional[float] = Field(None, description="Basement area sqft")
    LotArea: Optional[float] = Field(None, description="Lot size sqft")
    LotFrontage: Optional[float] = Field(None, description="Street-connected frontage feet")
    YearBuilt: Optional[int] = Field(None, description="Year constructed")
    OverallQual: Optional[int] = Field(None, ge=1, le=10, description="Quality 1-10")
    BedroomAbvGr: Optional[int] = Field(None, description="Bedrooms above grade")
    FullBath: Optional[int] = Field(None, description="Full bathrooms")
    HalfBath: Optional[int] = Field(None, description="Half bathrooms")
    GarageCars: Optional[int] = Field(None, description="Garage car capacity")
    GarageArea: Optional[float] = Field(None, description="Garage area sqft")

    # 1 categorical feature
    Neighborhood: Optional[str] = Field(None, description="Ames neighborhood name")

    # Completeness metadata
    extracted_features: list[str] = Field(default_factory=list)
    missing_features: list[str] = Field(default_factory=list)
    confidence: str = Field("insufficient", pattern="^(complete|partial|insufficient)$")
    follow_up_message: Optional[str] = None

    @property
    def all_feature_names(self) -> list[str]:
        return [
            "GrLivArea", "TotalBsmtSF", "LotArea", "LotFrontage", "YearBuilt",
            "OverallQual", "BedroomAbvGr", "FullBath", "HalfBath",
            "GarageCars", "GarageArea", "Neighborhood",
        ]

    def compute_completeness(self) -> None:
        """Recompute extracted/missing lists and confidence from current values."""
        self.extracted_features = []
        self.missing_features = []
        for name in self.all_feature_names:
            if getattr(self, name) is not None:
                self.extracted_features.append(name)
            else:
                self.missing_features.append(name)

        n = len(self.extracted_features)
        if n == 12:
            self.confidence = "complete"
        elif n >= 6:
            self.confidence = "partial"
        else:
            self.confidence = "insufficient"


# --- Combined Response ---

class PredictionResponse(BaseModel):
    """Pydantic schema #2 — full response returned to the client."""

    extracted_features: ExtractedFeatures
    predicted_price: float
    interpretation: str
    market_context: dict


# --- Conversational reply (when features are incomplete) ---

class FollowUpResponse(BaseModel):
    extracted_features: ExtractedFeatures
    assistant_message: str
