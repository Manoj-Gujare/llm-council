"""Shared web search (Tavily) for the LLM Council.

Every council member receives the exact same evidence. Search runs once,
before Stage 1, and models never search on their own.
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import httpx

from .openrouter import query_model
from .config import (
    TAVILY_API_KEY,
    TAVILY_API_URL,
    SEARCH_PLANNER_MODEL,
    SEARCH_MAX_QUERIES,
    SEARCH_RESULTS_PER_QUERY,
    SEARCH_MAX_SOURCES,
    SEARCH_DEPTH,
    SEARCH_SNIPPET_CHARS,
)


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


def normalize_query(query: str) -> str:
    """Lowercase and collapse whitespace."""
    return " ".join(query.lower().split())


def make_source(source_id: int, result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Build a numbered source entry from a Tavily result."""
    return {
        "id": source_id,
        "title": result["title"],
        "url": result["url"],
        "content": result["content"][:SEARCH_SNIPPET_CHARS],
        "published_date": result.get("published_date"),
        "query": query,
    }


def select_round_robin(
    result_lists: List[List[Dict[str, Any]]],
    limit: int,
    seen_urls: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Pick results round-robin across queries: best of query 1, best of query 2, ...
    then second-best of each, and so on. Scores are not comparable across queries,
    so we never sort globally. Dedupes by normalized URL.

    Returns:
        List of (result, query_index) pairs as dicts: {"result": ..., "query_index": ...}
    """
    seen = set() if seen_urls is None else seen_urls
    ranked = [
        sorted(results, key=lambda r: r.get("score") or 0, reverse=True)
        for results in result_lists
    ]
    picked = []
    depth = max((len(r) for r in ranked), default=0)
    for i in range(depth):
        for query_index, results in enumerate(ranked):
            if len(picked) >= limit:
                return picked
            if i >= len(results):
                continue
            key = normalize_url(results[i]["url"])
            if key in seen:
                continue
            seen.add(key)
            picked.append({"result": results[i], "query_index": query_index})
    return picked


def build_context(sources: List[Dict[str, Any]]) -> str:
    """Format numbered sources as plain text for prompts."""
    blocks = []
    for source in sources:
        published_line = ""
        if source.get("published_date"):
            published_line = f"\nPublished: {source['published_date']}"
        blocks.append(
            f"[{source['id']}] {source['title']}\nURL: {source['url']}{published_line}\n{source['content']}"
        )
    return "\n\n".join(blocks)


def evidence_block(context: str) -> str:
    """Wrap the search context; this exact block goes into every Stage 1/2/3 prompt."""
    return (
        "WEB SEARCH RESULTS (identical for every council member):\n\n"
        f"{context}\n\n"
        "END OF WEB SEARCH RESULTS"
    )


def extract_json(text: str, opener: str, closer: str) -> Optional[Any]:
    """Extract the first JSON value delimited by opener/closer from model output."""
    if not text:
        return None
    start = text.find(opener)
    end = text.rfind(closer)
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None


async def plan_search_queries(user_query: str) -> Optional[List[str]]:
    """
    Ask the planner model which web searches would help answer the question.

    Returns:
        List of up to SEARCH_MAX_QUERIES queries, [] if no search is needed,
        or None if the planner failed.
    """
    planner_prompt = f"""You decide which web searches would help answer a user's question.

Question: {user_query}

If the question needs current facts, specific data, recent events, or verifiable claims, return up to {SEARCH_MAX_QUERIES} short, distinct web search queries.
If web search would not help (for example pure coding, math, or creative writing), return an empty array.

Return ONLY a JSON array of strings, for example: ["query one", "query two"]"""

    messages = [{"role": "user", "content": planner_prompt}]
    response = await query_model(SEARCH_PLANNER_MODEL, messages, timeout=30.0)
    if response is None:
        return None

    parsed = extract_json(response.get("content") or "", "[", "]")
    if not isinstance(parsed, list):
        return None

    queries = [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    return queries[:SEARCH_MAX_QUERIES]


def empty_search(status: str) -> Dict[str, Any]:
    return {"status": status, "queries": [], "gap_queries": [], "sources": [], "context": ""}


async def run_stage0_search(user_query: str) -> Dict[str, Any]:
    """
    Stage 0: shared search. The planner and a search on the raw question run in
    parallel; planned queries are searched as soon as the planner returns.

    Returns:
        {"status": "ok" | "skipped" | "no_key" | "failed",
         "queries": [...], "gap_queries": [], "sources": [...], "context": str}
    """
    if not TAVILY_API_KEY:
        return empty_search("no_key")

    planner_task = asyncio.create_task(plan_search_queries(user_query))
    raw_task = asyncio.create_task(tavily_search(user_query))

    planned = await planner_task
    if planned == []:
        # Planner says no search needed; drop the speculative raw search.
        raw_task.cancel()
        return empty_search("skipped")

    queries = [user_query]
    if planned:
        used = {normalize_query(user_query)}
        for q in planned:
            if normalize_query(q) not in used:
                used.add(normalize_query(q))
                queries.append(q)

    extra_results = await asyncio.gather(*[tavily_search(q) for q in queries[1:]])
    raw_results = await raw_task
    result_lists = [raw_results] + list(extra_results)

    picked = select_round_robin(result_lists, SEARCH_MAX_SOURCES)
    sources = [
        make_source(i + 1, p["result"], queries[p["query_index"]])
        for i, p in enumerate(picked)
    ]

    status = "ok" if sources else "failed"
    return {
        "status": status,
        "queries": queries,
        "gap_queries": [],
        "sources": sources,
        "context": build_context(sources),
    }


def public_search(search: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Search info for the UI and storage: no prompt context, no snippet text."""
    if search is None:
        return None
    return {
        "status": search["status"],
        "queries": search.get("queries", []),
        "gap_queries": search.get("gap_queries", []),
        "sources": [
            {k: v for k, v in s.items() if k != "content"}
            for s in search.get("sources", [])
        ],
    }
