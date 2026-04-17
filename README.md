# AI Real Estate Agent

A conversational AI agent that predicts house prices in Ames, Iowa. You describe a property in plain English, and the agent extracts the relevant details, asks follow-up questions if anything is missing, runs a trained ML model, and explains the result — all through a chat interface.

## How It Works

The app uses a two-stage LLM prompt chain:

1. **Stage 1 – Feature Extraction**: The LLM parses the user's natural language description into structured features (bedrooms, square footage, neighborhood, etc.). If some details are missing, it asks follow-up questions conversationally.
2. **Stage 2 – Prediction Interpretation**: Once all features are collected, a Gradient Boosting model predicts the price. The LLM then explains the prediction in context — comparing it to neighborhood averages, highlighting what drives the price up or down.

The ML model was trained on the [Ames Housing dataset](http://jse.amstat.org/v19n3/decock.pdf) using 12 features (11 numeric + neighborhood). It achieves an R² of ~0.91 on the test set.

## Project Structure

```
├── main.py                  # FastAPI entry point
├── routers/
│   ├── chat.py              # /chat endpoint (prompt chain orchestrator)
│   ├── llm.py               # Gemini/Gemma API calls (Stage 1 + Stage 2)
│   ├── ml_model.py          # Load pipeline, run predictions
│   ├── prompts.py           # System prompts (V1 and V2)
│   ├── schemas.py           # Pydantic models
│   ├── security.py          # Input validation & prompt injection detection
│   └── config.py            # App settings
├── models/
│   ├── pipeline.joblib      # Trained sklearn pipeline
│   └── training_stats.json  # Training data summary for Stage 2
├── ui/
│   └── app.py               # Streamlit chat interface
├── ames-housing/
│   └── AmesHousing.csv      # Raw dataset
├── Dockerfile               # API container
├── docker-compose.yml       # API + UI services
└── pyproject.toml           # Dependencies (managed with uv)
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Google AI API key (for Gemini/Gemma models)

### Install & Run Locally

```bash
# Clone the repo
git clone https://github.com/moezys/AI-real-estate-agent.git
cd AI-real-estate-agent

# Install dependencies
uv sync

# Create .env file
cp .env.example .env
# Add your API key: GEMINI_API_KEY=your_key_here

# Start the API
uv run uvicorn main:app --reload

# In a separate terminal, start the UI
uv run streamlit run ui/app.py
```

The API runs on `http://localhost:8000` and the UI on `http://localhost:8501`.

### Run with Docker

```bash
docker compose up --build
```

## Tech Stack

- **ML**: scikit-learn (GradientBoostingRegressor with ColumnTransformer pipeline)
- **LLM**: Google Gemma 4 via the google-genai SDK
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Deployment**: Docker (Railway / Render)

## Features

- Conversational property description, no forms to fill out
- Asks follow-up questions when details are missing
- Prompt injection detection
- Two prompt versions (V1: structured template, V2: chain-of-thought) for experimentation
- Sidebar tracks which features have been extracted in real time
- Graceful fallback when the LLM is unavailable