"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-5.5",
    "x-ai/grok-4.7",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "google/gemini-3.1-pro-preview"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# Web search (Tavily) - a tool each council member may call during Stage 1
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_API_URL = "https://api.tavily.com/search"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


WEB_SEARCH_DEFAULT = _env_flag("WEB_SEARCH_DEFAULT", True)
SEARCH_MAX_CALLS_PER_MODEL = 3  # web_search tool calls allowed per model in Stage 1
SEARCH_RESULTS_PER_QUERY = 5
SEARCH_DEPTH = "basic"
SEARCH_SNIPPET_CHARS = 1000  # snippet length the searching model sees
REVIEW_SNIPPET_CHARS = 300  # shorter snippets for Stage 2 / Stage 3 prompts (cost)
