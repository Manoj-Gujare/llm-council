"""Web search (Tavily) as a tool for the LLM Council.

During Stage 1 each council member gets a `web_search` tool and decides for
itself whether, and what, to search. Every model keeps its own numbered
sources, which travel with its answer into Stage 2 and Stage 3.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Callable
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx

from .openrouter import query_model
from .config import (
    TAVILY_API_KEY,
    TAVILY_API_URL,
    SEARCH_MAX_CALLS_PER_MODEL,
    SEARCH_RESULTS_PER_QUERY,
    SEARCH_DEPTH,
    SEARCH_SNIPPET_CHARS,
)

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web. Returns numbered results (title, URL, date, snippet). "
            "Use it for current events, specific facts or figures, and claims worth verifying."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short, specific web search query.",
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_SYSTEM_PROMPT = f"""You have a web_search tool. You may call it up to {SEARCH_MAX_CALLS_PER_MODEL} times.
- Search when the question needs current facts, specific data, recent events, or claims worth verifying. Skip it when your own knowledge is enough (for example pure coding, math, or creative writing).
- Cite the search results you rely on as [n], using the numbers shown in the results.
- If sources conflict with each other or look outdated, say so.
- Never invent sources, source numbers, or URLs."""

SearchCallback = Callable[[str, str, int], None]


async def tavily_search(query: str) -> List[Dict[str, Any]]:
    """
    Run a single Tavily search.

    Returns:
        List of results (title, url, content, score, published_date).
        Returns [] on any error; never raises.
    """
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": SEARCH_RESULTS_PER_QUERY,
        "search_depth": SEARCH_DEPTH,
        "include_published_date": True,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(TAVILY_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results") or []:
            if not item.get("url"):
                continue
            results.append({
                "title": item.get("title") or item["url"],
                "url": item["url"],
                "content": item.get("content") or "",
                "score": item.get("score") or 0,
                "published_date": item.get("published_date"),
            })
        return results

    except Exception as e:
        print(f"Error running Tavily search for {query!r}: {e}")
        return []


def normalize_url(url: str) -> str:
    """Normalize a URL for dedupe: lowercase host, strip trailing slash and utm_* params."""
    try:
        parts = urlsplit(url.strip())
        query = urlencode([
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_")
        ])
        path = parts.path.rstrip("/")
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))
    except Exception:
        return url.strip().lower().rstrip("/")


def format_sources(sources: List[Dict[str, Any]], snippet_chars: Optional[int] = None) -> str:
    """Format numbered sources as plain text for prompts and tool results."""
    blocks = []
    for source in sources:
        published_line = ""
        if source.get("published_date"):
            published_line = f"\nPublished: {source['published_date']}"
        content = source.get("content") or ""
        if snippet_chars is not None:
            content = content[:snippet_chars]
        blocks.append(
            f"[{source['id']}] {source['title']}\nURL: {source['url']}{published_line}\n{content}"
        )
    return "\n\n".join(blocks)


def cited_sources(text: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sources whose [n] number appears in the text, e.g. [2] or [1, 3]."""
    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text or ""):
        cited.update(int(n) for n in group.split(","))
    return [s for s in sources if s["id"] in cited]


def public_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sources for the UI and storage: no snippet text."""
    return [{k: v for k, v in s.items() if k != "content"} for s in sources]


def parse_tool_query(tool_call: Dict[str, Any]) -> Optional[str]:
    """Extract the query argument from a web_search tool call, or None."""
    function = tool_call.get("function") or {}
    if function.get("name") != "web_search":
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    return " ".join(query.split())


class SourceBook:
    """One model's numbered sources. The same URL found twice keeps its first number."""

    def __init__(self):
        self.sources: List[Dict[str, Any]] = []
        self._by_url: Dict[str, Dict[str, Any]] = {}

    def add(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        key = normalize_url(result["url"])
        if key in self._by_url:
            return self._by_url[key]
        source = {
            "id": len(self.sources) + 1,
            "title": result["title"],
            "url": result["url"],
            "content": result["content"][:SEARCH_SNIPPET_CHARS],
            "published_date": result.get("published_date"),
            "query": query,
        }
        self.sources.append(source)
        self._by_url[key] = source
        return source


def tool_result_text(query: Optional[str], sources: List[Dict[str, Any]]) -> str:
    if query is None:
        return 'Invalid call. Use web_search with {"query": "..."}.'
    if not sources:
        return f"No results found for {query!r}."
    return format_sources(sources)


async def answer_with_search(
    model: str,
    user_query: str,
    on_search: Optional[SearchCallback] = None,
) -> Optional[Dict[str, Any]]:
    """
    Let one model answer with the web_search tool available.

    The model may call the tool up to SEARCH_MAX_CALLS_PER_MODEL times; after
    that it is asked to answer with tools disabled. If the very first request
    fails (for example the model rejects tools), retry once without tools.

    Returns:
        {"response": str, "sources": [...], "searches": [{"query", "source_ids"}]},
        plus "search_failed": True when it fell back to answering without the tool,
        or None if the model failed.
    """
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    book = SourceBook()
    searches: List[Dict[str, Any]] = []
    calls_used = 0

    # Each round with tool calls uses at least one call, so this always ends.
    for _ in range(SEARCH_MAX_CALLS_PER_MODEL + 1):
        limit_reached = calls_used >= SEARCH_MAX_CALLS_PER_MODEL
        response = await query_model(
            model,
            messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="none" if limit_reached else "auto",
        )

        if response is None:
            if calls_used == 0:
                # query_model already retried transient errors, so the tool request
                # itself failed; answer without search rather than drop the model
                print(f"Model {model} failed with the web_search tool; answering without search")
                fallback = await query_model(model, [{"role": "user", "content": user_query}])
                if fallback is None:
                    return None
                return {
                    "response": fallback.get("content") or "",
                    "sources": [],
                    "searches": [],
                    "search_failed": True,
                }
            return None

        tool_calls = response.get("tool_calls") or []
        if not tool_calls or limit_reached:
            return {
                "response": response.get("content") or "",
                "sources": book.sources,
                "searches": searches,
            }

        assistant_message = {
            "role": "assistant",
            "content": response.get("content") or "",
            "tool_calls": tool_calls,
        }
        if response.get("reasoning_details"):
            # Gemini and Claude need their reasoning passed back across tool turns
            assistant_message["reasoning_details"] = response["reasoning_details"]
        messages.append(assistant_message)

        planned = []  # (tool_call, query or None, run search?)
        for tool_call in tool_calls:
            query = parse_tool_query(tool_call)
            if calls_used >= SEARCH_MAX_CALLS_PER_MODEL:
                planned.append((tool_call, query, False))
                continue
            calls_used += 1
            planned.append((tool_call, query, query is not None))

        result_lists = await asyncio.gather(*[
            tavily_search(query) if run else asyncio.sleep(0, result=[])
            for _, query, run in planned
        ])

        for (tool_call, query, run), results in zip(planned, result_lists):
            if run:
                found = [book.add(r, query) for r in results]
                searches.append({"query": query, "source_ids": [s["id"] for s in found]})
                if on_search is not None:
                    on_search(model, query, len(found))
                content = tool_result_text(query, found)
            elif query is None:
                content = tool_result_text(None, [])
            else:
                content = "Search limit reached. Answer with the results you already have."
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.get("id"),
                "content": content,
            })

    return None
