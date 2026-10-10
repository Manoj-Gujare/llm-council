"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Web search (Tavily) - shared evidence for every council member
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/search"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


WEB_SEARCH_DEFAULT = _env_flag("WEB_SEARCH_DEFAULT", True)
SEARCH_PLANNER_MODEL = "google/gemini-2.5-flash"
SEARCH_MAX_QUERIES = 3
SEARCH_RESULTS_PER_QUERY = 5
SEARCH_MAX_SOURCES = 8  # Stage 0 pool
SEARCH_DEPTH = "basic"
SEARCH_SNIPPET_CHARS = 1000

# Stage 0.5 gap-fill: models may request extra queries for the shared pool
GAP_FILL_ENABLED = _env_flag("GAP_FILL_ENABLED", True)
GAP_MAX_QUERIES_PER_MODEL = 2
GAP_MAX_TOTAL_QUERIES = 6
GAP_SOURCES_PER_QUERY = 2
SEARCH_MAX_SOURCES_TOTAL = 12  # hard cap on the final pool
