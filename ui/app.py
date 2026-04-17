"""
Streamlit chat UI for the AI Real Estate Agent.

Connects to the FastAPI backend at /chat.
Shows a conversational interface with feature tracking sidebar.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏠", layout="wide")
st.title("🏠 AI Real Estate Agent — Ames, Iowa")
st.caption("Describe a property in plain English and I'll predict its price.")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "features" not in st.session_state:
    st.session_state.features = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# --- Sidebar: feature tracker ---
# User-friendly display names
DISPLAY_NAMES = {
    "GrLivArea": "Living Area (sqft)",
    "TotalBsmtSF": "Basement Area (sqft)",
    "LotArea": "Lot Size (sqft)",
    "LotFrontage": "Street Frontage (ft)",
    "YearBuilt": "Year Built",
    "OverallQual": "Quality (1-10)",
    "BedroomAbvGr": "Bedrooms",
    "FullBath": "Full Bathrooms",
    "HalfBath": "Half Bathrooms",
    "GarageCars": "Garage Capacity (cars)",
    "GarageArea": "Garage Area (sqft)",
    "Neighborhood": "Neighborhood",
}

with st.sidebar:
    st.header("📋 Extracted Features")

    if st.session_state.features:
        feats = st.session_state.features
        all_names = list(DISPLAY_NAMES.keys())
        for name in all_names:
            display = DISPLAY_NAMES[name]
            val = feats.get(name)
            if val is not None:
                st.markdown(f"✅ **{display}:** {val}")
            else:
                st.markdown(f"❌ **{display}:** _missing_")

        extracted = feats.get("extracted_features", [])
        missing = feats.get("missing_features", [])
        st.divider()
        st.metric("Extracted", f"{len(extracted)} / 12")
        st.progress(len(extracted) / 12)
    else:
        st.info("Start chatting to see features appear here.")

    st.divider()
    if st.button("🔄 New Conversation"):
        st.session_state.messages = []
        st.session_state.features = None
        st.session_state.prediction = None
        st.rerun()


# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Show prediction result if available ---
if st.session_state.prediction:
    pred = st.session_state.prediction
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("💰 Predicted Price", f"${pred['predicted_price']:,.0f}")
    with col2:
        ctx = pred.get("market_context", {})
        if ctx:
            pct = ctx.get("price_vs_median_pct", 0)
            direction = "above" if pct > 0 else "below"
            st.caption(
                f"{abs(pct):.1f}% {direction} the Ames median "
                f"(${ctx.get('ames_median_price', 0):,.0f})"
            )

    with st.expander("📊 Full Interpretation", expanded=True):
        st.markdown(pred.get("interpretation", ""))


# --- Chat input ---
if user_input := st.chat_input("Describe a property..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build request payload
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_URL}/chat",
                json={"messages": api_messages},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            data = None
            st.error("Cannot connect to the API. Is the FastAPI server running?")
        except requests.exceptions.HTTPError as e:
            data = None
            st.error(f"API error: {e}")
        except Exception as e:
            data = None
            st.error(f"Unexpected error: {e}")

    if data:
        # Update features in sidebar
        st.session_state.features = data.get("extracted_features", {})

        # Check if it's a follow-up or final prediction
        if "predicted_price" in data:
            # Final prediction
            st.session_state.prediction = data
            assistant_msg = (
                f"**Predicted Price: ${data['predicted_price']:,.0f}**\n\n"
                "See the detailed interpretation below."
            )
        else:
            # Follow-up question
            assistant_msg = data.get("assistant_message", "Could you tell me more about the property?")

        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
        st.rerun()
