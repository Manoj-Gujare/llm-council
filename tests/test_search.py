"""Tests for the web_search tool in Stage 1 and source handling in Stages 2-3. Fully mocked: no network."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend import council, search, storage
from backend.config import COUNCIL_MODELS, SEARCH_MAX_CALLS_PER_MODEL


def run(coro):
    return asyncio.run(coro)


def result(url, score=0.5, title=None):
    return {
        "title": title or url,
        "url": url,
        "content": f"content of {url}",
        "score": score,
        "published_date": None,
    }


def tool_call(query, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": query})},
    }


class ScriptedModel:
    """Fake search.query_model: returns scripted replies in order, records every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def __call__(self, model, messages, timeout=120.0, tools=None, tool_choice=None):
        self.calls.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "tool_choice": tool_choice,
        })
        reply = self.replies.pop(0)
        return reply(self.calls[-1]) if callable(reply) else reply


class FakeTavily:
    """Fake tavily_search: two results per query, URLs derived from the query."""

    def __init__(self, results=None):
        self.queries = []
        self.results = results

    async def __call__(self, query):
        self.queries.append(query)
        if self.results is not None:
            return self.results
        slug = query.replace(" ", "-")
        return [result(f"https://{slug}.com/a"), result(f"https://{slug}.com/b")]


def answer_with(fake_model, tavily=None, on_search=None):
    tavily = tavily or FakeTavily()
    with patch.object(search, "query_model", fake_model), \
         patch.object(search, "tavily_search", tavily):
        return run(search.answer_with_search("m1", "What is new?", on_search)), tavily


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_normalize_url_dedupes_case_slash_and_utm():
    a = search.normalize_url("https://Example.COM/path/?utm_source=x&id=3&utm_medium=y")
    b = search.normalize_url("https://example.com/path?id=3")
    assert a == b


def test_cited_sources_parses_single_and_grouped_citations():
    sources = [{"id": i} for i in range(1, 6)]
    cited = search.cited_sources("A [1], B [3, 4] and [9] and [x]", sources)
    assert [s["id"] for s in cited] == [1, 3, 4]
    assert search.cited_sources("no citations", sources) == []


def test_parse_tool_query():
    assert search.parse_tool_query(tool_call("  rust   2026 ")) == "rust 2026"
    assert search.parse_tool_query({"function": {"name": "other", "arguments": "{}"}}) is None
    assert search.parse_tool_query({"function": {"name": "web_search", "arguments": "not json"}}) is None
    assert search.parse_tool_query({"function": {"name": "web_search", "arguments": '{"query": ""}'}}) is None


def test_tavily_search_never_raises():
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    with patch.object(search.httpx, "AsyncClient", Boom):
        assert run(search.tavily_search("q")) == []


# ---------------------------------------------------------------------------
# answer_with_search: the tool loop
# ---------------------------------------------------------------------------

def test_model_answers_without_searching():
    fake = ScriptedModel({"content": "2 + 2 = 4"})
    out, tavily = answer_with(fake)
    assert out == {"response": "2 + 2 = 4", "sources": [], "searches": []}
    assert tavily.queries == []
    call = fake.calls[0]
    assert call["tools"] == [search.WEB_SEARCH_TOOL] and call["tool_choice"] == "auto"
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {"role": "user", "content": "What is new?"}


def test_one_search_round_then_answer():
    fake = ScriptedModel(
        {"content": "", "tool_calls": [tool_call("rust news")], "reasoning_details": [{"r": 1}]},
        {"content": "Rust shipped [2]."},
    )
    seen = []
    out, tavily = answer_with(fake, on_search=lambda m, q, n: seen.append((m, q, n)))

    assert tavily.queries == ["rust news"]
    assert out["response"] == "Rust shipped [2]."
    assert [s["id"] for s in out["sources"]] == [1, 2]
    assert out["sources"][0]["query"] == "rust news"
    assert out["searches"] == [{"query": "rust news", "source_ids": [1, 2]}]
    assert seen == [("m1", "rust news", 2)]

    second = fake.calls[1]["messages"]
    assistant, tool = second[2], second[3]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["reasoning_details"] == [{"r": 1}]
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_1"
    assert "[1] https://rust-news.com/a" in tool["content"]
    assert "[2] https://rust-news.com/b" in tool["content"]


def test_search_cap_forces_final_answer():
    keeps_searching = [
        {"content": "", "tool_calls": [tool_call(f"q{i}", f"c{i}")]}
        for i in range(SEARCH_MAX_CALLS_PER_MODEL)
    ]
    fake = ScriptedModel(*keeps_searching, {"content": "final answer"})
    out, tavily = answer_with(fake)

    assert len(tavily.queries) == SEARCH_MAX_CALLS_PER_MODEL
    assert out["response"] == "final answer"
    choices = [c["tool_choice"] for c in fake.calls]
    assert choices == ["auto"] * SEARCH_MAX_CALLS_PER_MODEL + ["none"]


def test_extra_parallel_calls_over_cap_are_refused():
    calls = [tool_call(f"q{i}", f"c{i}") for i in range(SEARCH_MAX_CALLS_PER_MODEL + 2)]
    fake = ScriptedModel({"content": "", "tool_calls": calls}, {"content": "done"})
    out, tavily = answer_with(fake)

    assert len(tavily.queries) == SEARCH_MAX_CALLS_PER_MODEL
    tool_messages = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == len(calls)  # every call gets an answer
    assert "Search limit reached" in tool_messages[-1]["content"]
    assert fake.calls[1]["tool_choice"] == "none"
    assert out["response"] == "done"


def test_same_url_keeps_first_source_number():
    fake = ScriptedModel(
        {"content": "", "tool_calls": [tool_call("a", "c1")]},
        {"content": "", "tool_calls": [tool_call("b", "c2")]},
        {"content": "ok"},
    )
    tavily = FakeTavily(results=[result("https://same.com/x"), result("https://SAME.com/x/?utm_source=y")])
    out, _ = answer_with(fake, tavily)
    assert len(out["sources"]) == 1
    assert out["searches"] == [{"query": "a", "source_ids": [1, 1]}, {"query": "b", "source_ids": [1, 1]}]


def test_invalid_tool_call_gets_error_message_and_counts():
    bad = {"id": "bad", "function": {"name": "web_search", "arguments": "{}"}}
    fake = ScriptedModel({"content": "", "tool_calls": [bad]}, {"content": "ok"})
    out, tavily = answer_with(fake)
    assert tavily.queries == []
    tool = fake.calls[1]["messages"][-1]
    assert tool["tool_call_id"] == "bad" and "Invalid call" in tool["content"]
    assert out["response"] == "ok"


def test_no_results_message():
    fake = ScriptedModel({"content": "", "tool_calls": [tool_call("x")]}, {"content": "ok"})
    out, _ = answer_with(fake, FakeTavily(results=[]))
    assert "No results found" in fake.calls[1]["messages"][-1]["content"]
    assert out["sources"] == []


def test_first_request_failure_retries_without_tools():
    fake = ScriptedModel(None, {"content": "plain answer"})
    out, _ = answer_with(fake)
    assert out == {"response": "plain answer", "sources": [], "searches": [], "search_failed": True}
    assert fake.calls[1]["tools"] is None
    assert fake.calls[1]["messages"] == [{"role": "user", "content": "What is new?"}]


def test_failure_after_searching_returns_none():
    fake = ScriptedModel({"content": "", "tool_calls": [tool_call("x")]}, None)
    out, _ = answer_with(fake)
    assert out is None


def test_both_requests_failing_returns_none():
    out, _ = answer_with(ScriptedModel(None, None))
    assert out is None


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def test_stage1_with_search_gives_each_model_its_own_loop():
    async def fake_answer(model, query, on_search=None):
        if model == COUNCIL_MODELS[1]:
            return None
        return {"response": f"{model} [1]", "sources": [{"id": 1, "url": "u", "content": "c"}], "searches": []}

    with patch.object(council, "answer_with_search", fake_answer):
        out = run(council.stage1_collect_responses("Q?", web_search=True))
    assert [r["model"] for r in out] == [m for m in COUNCIL_MODELS if m != COUNCIL_MODELS[1]]
    assert all(r["sources"][0]["id"] == 1 for r in out)


def test_stage1_without_search_uses_plain_prompt():
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec):
        out = run(council.stage1_collect_responses("Q?"))
    assert set(rec.calls[0].values()) == {"Q?"}
    assert all("sources" not in r for r in out)


# ---------------------------------------------------------------------------
# Stages 2-3: per-response sources
# ---------------------------------------------------------------------------

class Recorder:
    """Fake query_models_parallel that records every prompt per model."""

    def __init__(self, reply="ok"):
        self.calls = []  # list of {model: prompt}
        self.reply = reply

    async def __call__(self, models, messages, timeout=120.0):
        prompt = messages[0]["content"]
        self.calls.append({m: prompt for m in models})
        reply = self.reply
        if callable(reply):
            return {m: reply(m, prompt) for m in models}
        return {m: {"content": reply} for m in models}


def stage_reply(model, prompt):
    if "FINAL RANKING:" in prompt:
        return {"content": "Good.\n\nFINAL RANKING:\n1. Response A\n2. Response B"}
    return {"content": "an answer"}


def src(i, url):
    return {"id": i, "title": f"T{i}", "url": url, "content": "x" * 2000, "published_date": None, "query": "q"}


SEARCHED_STAGE1 = [
    {"model": "m1", "response": "Claim one [2].", "sources": [src(1, "https://a.com"), src(2, "https://b.com")], "searches": []},
    {"model": "m2", "response": "No search here.", "sources": [], "searches": []},
]


def test_stage2_lists_only_cited_sources_per_response():
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec):
        run(council.stage2_collect_rankings("Q?", SEARCHED_STAGE1))
    prompt = next(iter(rec.calls[0].values()))

    assert "Sources cited by Response A:\n[2] T2\nURL: https://b.com" in prompt
    assert "https://a.com" not in prompt  # retrieved but not cited
    assert "Sources cited by Response B" not in prompt
    assert "x" * (council.REVIEW_SNIPPET_CHARS + 1) not in prompt  # snippets trimmed
    assert "Fact-check claims against those sources" in prompt
    for model in ("m1", "m2"):
        assert model not in prompt  # still anonymous


def test_stage3_gets_sources_and_link_instructions():
    chair = AsyncMock(return_value={"content": "final"})
    with patch.object(council, "query_model", chair):
        run(council.stage3_synthesize_final("Q?", SEARCHED_STAGE1, [{"model": "m1", "ranking": "R"}]))
    prompt = chair.call_args.args[1][0]["content"]
    assert "Sources cited by this response:\n[2] T2" in prompt
    assert "markdown links [title](url)" in prompt


def test_prompts_without_citations_match_original_code_exactly():
    """With no cited sources, prompts must equal what the pre-search code produced."""
    stage1 = [
        {"model": "m1", "response": "r1"},
        {"model": "m2", "response": "r2", "sources": [src(1, "https://a.com")], "searches": []},
    ]
    rec = Recorder(stage_reply)
    chair = AsyncMock(return_value={"content": "x"})
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", chair):
        run(council.stage2_collect_rankings("Q?", stage1))
        run(council.stage3_synthesize_final("Q?", stage1, [{"model": "m1", "ranking": "R"}]))

    expected_stage2 = """You are evaluating different responses to the following question:

Question: Q?

Here are the responses from different models (anonymized):

Response A:
r1

Response B:
r2

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...
Response C offers the most comprehensive answer...

FINAL RANKING:
1. Response C
2. Response A
3. Response B

Now provide your evaluation and ranking:"""

    expected_stage3 = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: Q?

STAGE 1 - Individual Responses:
Model: m1
Response: r1

Model: m2
Response: r2

STAGE 2 - Peer Rankings:
Model: m1
Ranking: R

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

    assert next(iter(rec.calls[0].values())) == expected_stage2
    assert chair.call_args.args[1][0]["content"] == expected_stage3


# ---------------------------------------------------------------------------
# Full council
# ---------------------------------------------------------------------------

def run_council(web_search, key="tvly-test"):
    async def fake_answer(model, query, on_search=None):
        return {"response": "cited [1]", "sources": [src(1, f"https://{model}.com")], "searches": [{"query": "q", "source_ids": [1]}]}

    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", AsyncMock(return_value={"content": "final"})), \
         patch.object(council, "answer_with_search", fake_answer), \
         patch.object(council, "TAVILY_API_KEY", key):
        return run(council.run_full_council("Q?", web_search=web_search)), rec


def test_full_council_with_search_returns_public_sources():
    (s1, s2, s3, metadata), rec = run_council(True)
    assert metadata["search"] == {"status": "tool"}
    assert len(s1) == len(COUNCIL_MODELS)
    assert all("content" not in s for r in s1 for s in r["sources"])
    # Stage 2 prompt still got the snippets
    assert "Sources cited by Response A" in next(iter(rec.calls[0].values()))
    assert s3["response"] == "final"


@pytest.mark.parametrize("web_search, key, status", [
    (False, "tvly-test", None),
    (True, None, {"status": "no_key"}),
])
def test_full_council_plain_when_search_off_or_no_key(web_search, key, status):
    (s1, _, s3, metadata), rec = run_council(web_search, key)
    assert metadata["search"] == status
    assert set(rec.calls[0].values()) == {"Q?"}  # plain Stage 1 prompt
    assert all("sources" not in r for r in s1)
    assert s3["response"] == "final"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main

    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "generate_conversation_title", AsyncMock(return_value="Title"))
    return TestClient(main.app)


def searching_answer(model, query, on_search=None):
    async def go():
        if on_search is not None:
            on_search(model, "q " + model, 1)
        return {"response": "cited [1]", "sources": [src(1, "https://a.com")], "searches": [{"query": "q", "source_ids": [1]}]}
    return go()


def stream_events(client, body, key="tvly-test"):
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", AsyncMock(return_value={"content": "final"})), \
         patch.object(council, "answer_with_search", searching_answer), \
         patch.object(council, "TAVILY_API_KEY", key):
        conv = client.post("/api/conversations", json={}).json()
        resp = client.post(f"/api/conversations/{conv['id']}/message/stream", json=body)
        events = [
            json.loads(line[6:]) for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        saved = client.get(f"/api/conversations/{conv['id']}").json()
    return events, saved


def test_sse_search_on_streams_progress_per_model(client):
    events, saved = stream_events(client, {"content": "Q?", "web_search": True})
    types = [e["type"] for e in events]
    n = len(COUNCIL_MODELS)
    assert types == (
        ["stage1_start"] + ["search_progress"] * n
        + ["stage1_complete", "stage2_start", "stage2_complete",
           "stage3_start", "stage3_complete", "title_complete", "complete"]
    )
    assert events[0]["search"] == {"status": "tool"}
    assert {e["data"]["model"] for e in events[1:1 + n]} == set(COUNCIL_MODELS)

    stage1 = events[1 + n]["data"]
    assert all("content" not in s for r in stage1 for s in r["sources"])

    msg = saved["messages"][-1]
    assert msg["search"] == {"status": "tool"}
    assert msg["stage1"][0]["sources"][0]["url"] == "https://a.com"
    assert "content" not in msg["stage1"][0]["sources"][0]


def test_sse_search_off(client):
    events, saved = stream_events(client, {"content": "Q?", "web_search": False})
    types = [e["type"] for e in events]
    assert types == [
        "stage1_start", "stage1_complete", "stage2_start", "stage2_complete",
        "stage3_start", "stage3_complete", "title_complete", "complete",
    ]
    assert events[0]["search"] is None
    assert "search" not in saved["messages"][-1]


def test_sse_no_key_still_completes(client):
    events, saved = stream_events(client, {"content": "Q?", "web_search": True}, key=None)
    assert events[0]["search"] == {"status": "no_key"}
    assert "search_progress" not in [e["type"] for e in events]
    assert events[-1]["type"] == "complete"
    assert saved["messages"][-1]["search"] == {"status": "no_key"}


def test_non_streaming_endpoint_persists_sources(client):
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", AsyncMock(return_value={"content": "final"})), \
         patch.object(council, "answer_with_search", searching_answer), \
         patch.object(council, "TAVILY_API_KEY", "tvly-test"):
        conv = client.post("/api/conversations", json={}).json()
        data = client.post(f"/api/conversations/{conv['id']}/message", json={"content": "Q?"}).json()
        saved = client.get(f"/api/conversations/{conv['id']}").json()
    assert data["metadata"]["search"] == {"status": "tool"}
    assert saved["messages"][-1]["stage1"][0]["sources"][0]["id"] == 1


def test_old_conversation_without_search_loads(client):
    conv = client.post("/api/conversations", json={}).json()
    raw = storage.get_conversation(conv["id"])
    raw["messages"] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "stage1": [], "stage2": [], "stage3": {"model": "x", "response": "y"}},
    ]
    storage.save_conversation(raw)
    loaded = client.get(f"/api/conversations/{conv['id']}")
    assert loaded.status_code == 200
    assert "search" not in loaded.json()["messages"][1]
