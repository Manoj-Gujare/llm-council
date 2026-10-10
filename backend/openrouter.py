"""OpenRouter API client for making LLM requests."""

import asyncio
import random
import httpx
from typing import List, Dict, Any, Optional
from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    OPENROUTER_MAX_RETRIES,
    OPENROUTER_RETRY_BACKOFF,
    OPENROUTER_MAX_RETRY_AFTER,
)

# Transient errors per OpenRouter docs: timeout, rate limit, provider down/unavailable
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def retry_delay(attempt: int, retry_after: Optional[str]) -> float:
    """Seconds to wait before the next try: Retry-After if given, else exponential backoff with jitter."""
    try:
        seconds = float(retry_after) if retry_after else 0.0
    except ValueError:
        seconds = 0.0
    if seconds > 0:
        return min(seconds, OPENROUTER_MAX_RETRY_AFTER)
    return OPENROUTER_RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)


def describe_error(status: int, body: Any) -> str:
    """Status plus OpenRouter's error message, error_type and provider, when present."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return f"HTTP {status}"
    metadata = error.get("metadata") or {}
    details = [
        f"{key}={metadata[key]}"
        for key in ("error_type", "provider_name", "provider_code")
        if metadata.get(key)
    ]
    suffix = f" ({', '.join(details)})" if details else ""
    return f"HTTP {status}: {error.get('message', '')}{suffix}"


async def query_model(
    model: str,
    messages: List[Dict[str, Any]],
    timeout: float = 120.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        tools: Optional tool definitions (OpenAI function-calling format)
        tool_choice: Optional "auto" / "none"

    Returns:
        Response dict with 'content', optional 'reasoning_details' and
        optional 'tool_calls', or None if failed. Transient errors (rate
        limits, provider outages) are retried with backoff first.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        # Route only to providers that support tool calling, never silently drop the tools
        payload["provider"] = {"require_parameters": True}
    if tool_choice:
        payload["tool_choice"] = tool_choice

    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        can_retry = attempt < OPENROUTER_MAX_RETRIES
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload
                )

            try:
                data = response.json()
            except ValueError:
                data = None

            # Errors can also arrive inside a 200 response
            status = response.status_code
            if status == 200 and isinstance(data, dict) and data.get("error"):
                status = (data["error"].get("code") if isinstance(data["error"], dict) else None) or 502

            if status != 200 or not isinstance(data, dict):
                error = describe_error(status, data)
                if status in RETRY_STATUSES and can_retry:
                    delay = retry_delay(attempt, response.headers.get("Retry-After"))
                    print(f"Model {model} failed with {error}; retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                print(f"Error querying model {model}: {error}")
                return None

            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details'),
                'tool_calls': message.get('tool_calls')
            }

        except httpx.TransportError as e:
            # Connection problems are retried; a read timeout means the model is just slow
            if can_retry and not isinstance(e, httpx.ReadTimeout):
                delay = retry_delay(attempt, None)
                print(f"Model {model} connection error {e!r}; retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            print(f"Error querying model {model}: {e!r}")
            return None

        except Exception as e:
            print(f"Error querying model {model}: {e}")
            return None

    return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        timeout: Request timeout in seconds for each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(model, messages, timeout) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
