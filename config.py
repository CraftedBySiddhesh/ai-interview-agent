"""
config.py — Constants, API validation, and cached client initialisation.

Responsibilities (FR-071, FR-080, FR-081):
  - Centralise all constants (no magic strings in other modules)
  - Validate required environment variables at import time
  - Provide lru_cache-wrapped factory functions for Mistral and Tavily clients
  - Phase 2 note: lru_cache is easily swapped for FastAPI dependency injection
  - LLM swap note: to replace Mistral with Claude, change get_model() only —
    all LCEL chains and parsers remain unchanged.
"""

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Interview constants (FR-032, FR-075, FR-045, FR-003)
# ---------------------------------------------------------------------------
TOTAL_QUESTIONS: int = 8
MAX_HINTS: int = 3
DEFAULT_DIFFICULTY: str = "Medium"
INTERVIEW_TYPES: list[str] = ["Technical", "Behavioural", "Mixed"]
DIFFICULTY_LEVELS: list[str] = ["Easy", "Medium", "Hard"]
CODING_LANGUAGES: list[str] = [
    "Python", "JavaScript", "Java", "C#", "C++", "TypeScript", "Go", "SQL"
]

# ---------------------------------------------------------------------------
# Document processing constants (FR-014, FR-024, FR-025)
# ---------------------------------------------------------------------------
MAX_DOC_CHARS: int = 8_000          # hard truncation per document (FR-014)
CHUNK_SIZE: int = 500               # RAG chunk size in characters (FR-024/025)
CHUNK_OVERLAP: int = 50             # overlap between adjacent chunks (FR-024/025)
RAG_TOP_K: int = 3                  # chunks retrieved per index per turn (FR-028)

# ---------------------------------------------------------------------------
# Token guard constants (FR-075)
# ---------------------------------------------------------------------------
TOKEN_GUARD_WORDS: int = 6_000      # max total words before trimming
TOKEN_GUARD_KEEP_MESSAGES: int = 8  # messages preserved after trim (excl. SystemMessage)

# ---------------------------------------------------------------------------
# Adaptive difficulty constants (FR-055, FR-056, FR-057)
# ---------------------------------------------------------------------------
ADAPTIVE_AVG_THRESHOLD: float = 4.0
ADAPTIVE_LOOKBACK: int = 3

# ---------------------------------------------------------------------------
# Mistral model config (FR section 5.1)
# ---------------------------------------------------------------------------
MISTRAL_MODEL: str = "mistral-small-2506"
MISTRAL_TEMPERATURE: float = 0.7

# ---------------------------------------------------------------------------
# Tavily search config (FR-020, section 5.2)
# ---------------------------------------------------------------------------
TAVILY_MAX_RESULTS: int = 5
TAVILY_SNIPPET_CHARS: int = 300

# ---------------------------------------------------------------------------
# Required env key names
# ---------------------------------------------------------------------------
REQUIRED_ENV_KEYS: list[str] = ["MISTRAL_API_KEY", "TAVILY_API_KEY"]


# ---------------------------------------------------------------------------
# FR-071 — Validate env keys
# Called from app.py at startup before any UI is rendered.
# Returns list of missing key names (empty list = all present).
# ---------------------------------------------------------------------------
def validate_env_keys() -> list[str]:
    """Return names of any missing required environment variables."""
    missing = [k for k in REQUIRED_ENV_KEYS if not os.getenv(k)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
    else:
        logger.info("All required environment variables present.")
    return missing


# ---------------------------------------------------------------------------
# FR-080 — Cached Mistral client
# Phase 2 swap note: replace ChatMistralAI with ChatAnthropic here only.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_model():
    """Return a cached ChatMistralAI instance (mistral-small-2506)."""
    from langchain_mistralai import ChatMistralAI  # deferred import keeps config.py lightweight

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY is not set.")

    model = ChatMistralAI(
        model=MISTRAL_MODEL,
        temperature=MISTRAL_TEMPERATURE,
        mistral_api_key=api_key,
    )
    logger.info("Mistral model initialised: %s", MISTRAL_MODEL)
    return model


# ---------------------------------------------------------------------------
# FR-080 — Cached Tavily client
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_tavily():
    """Return a cached TavilyClient instance."""
    from tavily import TavilyClient  # deferred import

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError("TAVILY_API_KEY is not set.")

    client = TavilyClient(api_key=api_key)
    logger.info("Tavily client initialised.")
    return client
