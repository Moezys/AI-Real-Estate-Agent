import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()  # Load .env before anything reads env vars

from routers.chat import router as chat_router
from routers.ml_model import load_model

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load ML pipeline + training stats once
    load_model()
    logging.info("ML pipeline and training stats loaded.")
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="AI Real Estate Agent",
    description="Natural language property descriptions → price predictions for Ames, Iowa",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
