from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

GEMINI_MODEL = "gemma-4-31b-it"
DEFAULT_PROMPT_VERSION = "v2"
MAX_CONVERSATION_ROUNDS = 5
