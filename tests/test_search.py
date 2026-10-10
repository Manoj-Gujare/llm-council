"""Tests for shared web search and gap-fill. Fully mocked: no network."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend import council, search, storage
from backend.config import COUNCIL_MODELS


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


def source(i, url):
    return {"id": i, "title": url, "url": url, "content": f"content {i}", "published_date": None, "query": "q"}


def ok_search(n=3, queries=("question",)):
    sources = [source(i + 1, f"https://site{i + 1}.com/page") for i in range(n)]
    return {
        "status": "ok",
        "queries": list(queries),
        "gap_queries": [],
        "sources": sources,
        "context": search.build_context(sources),
    }


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
    return {"content": "an answer citing [1]"}


# ---------------------------------------------------------------------------
# URL dedupe and round-robin selection
# ---------------------------------------------------------------------------

def test_normalize_url_dedupes_case_slash_and_utm():
    a = search.normalize_url("https://Example.COM/path/?utm_source=x&id=3&utm_medium=y")
    b = search.normalize_url("https://example.com/path?id=3")
    assert a == b


def test_round_robin_includes_every_query():
    q1 = [result(f"https://a.com/{i}", score=0.99 - i * 0.01) for i in range(5)]
    q2 = [result(f"https://b.com/{i}", score=0.20 - i * 0.01) for i in range(5)]
    q3 = [result(f"https://c.com/{i}", score=0.10 - i * 0.01) for i in range(5)]
    picked = search.select_round_robin([q1, q2, q3], 6)
    hosts = [p["result"]["url"].split("/")[2] for p in picked]
    assert hosts == ["a.com", "b.com", "c.com", "a.com", "b.com", "c.com"]
    # Best-scored of each query comes first
    assert picked[0]["result"]["url"] == "https://a.com/0"


def test_round_robin_dedupes_urls_across_queries():
    q1 = [result("https://same.com/x/", 0.9), result("https://a.com/2", 0.5)]
    q2 = [result("https://SAME.com/x?utm_campaign=z", 0.9), result("https://b.com/2", 0.5)]
    picked = search.select_round_robin([q1, q2], 10)
    urls = [p["result"]["url"] for p in picked]
    assert len(urls) == 3
    assert "https://a.com/2" in urls and "https://b.com/2" in urls


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------

def test_stage0_merges_planner_and_raw_results():
    async def fake_tavily(q):
        return [result(f"https://{q.replace(' ', '-')}.com/{i}", 0.9 - i * 0.1) for i in range(5)]

    with patch.object(search, "TAVILY_API_KEY", "key"), \
         patch.object(search, "plan_search_queries", AsyncMock(return_value=["topic one", "topic two"])), \
         patch.object(search, "tavily_search", side_effect=fake_tavily) as tav:
        out = run(search.run_stage0_search("the question"))

    assert out["status"] == "ok"
    assert out["queries"] == ["the question", "topic one", "topic two"]
    assert tav.call_count == 3
    assert [s["id"] for s in out["sources"]] == list(range(1, 9))
    assert {s["query"] for s in out["sources"]} == set(out["queries"])
    assert "[1]" in out["context"] and "[8]" in out["context"]


def test_stage0_planner_failure_uses_raw_results_only():
    raw = [result(f"https://raw.com/{i}") for i in range(3)]
    with patch.object(search, "TAVILY_API_KEY", "key"), \
         patch.object(search, "plan_search_queries", AsyncMock(return_value=None)), \
         patch.object(search, "tavily_search", AsyncMock(return_value=raw)) as tav:
        out = run(search.run_stage0_search("q"))
    assert out["status"] == "ok"
    assert out["queries"] == ["q"]
    assert tav.call_count == 1
    assert len(out["sources"]) == 3


def test_stage0_planner_empty_skips_search():
    completed = []

    async def slow_tavily(q):
        await asyncio.sleep(5)
        completed.append(q)
        return [result("https://x.com")]

    with patch.object(search, "TAVILY_API_KEY", "key"), \
         patch.object(search, "plan_search_queries", AsyncMock(return_value=[])), \
         patch.object(search, "tavily_search", side_effect=slow_tavily) as tav:
        out = run(search.run_stage0_search("write a poem"))

    assert out["status"] == "skipped"
    assert out["sources"] == []
    # Only the speculative raw-question search was started, and it was cancelled
    # before finishing: no Tavily search completed and no planned queries ran.
    assert completed == []
    assert tav.call_count <= 1


def test_stage0_no_key():
    tav = AsyncMock()
    with patch.object(search, "TAVILY_API_KEY", None), \
         patch.object(search, "tavily_search", tav):
        out = run(search.run_stage0_search("q"))
    assert out["status"] == "no_key"
    tav.assert_not_called()


def test_stage0_all_searches_fail_status_failed():
    with patch.object(search, "TAVILY_API_KEY", "key"), \
         patch.object(search, "plan_search_queries", AsyncMock(return_value=["a b", "c d"])), \
         patch.object(search, "tavily_search", AsyncMock(return_value=[])):
        out = run(search.run_stage0_search("q"))
    assert out["status"] == "failed"
    assert out["queries"] == ["q", "a b", "c d"]
    assert council.get_search_context(out) is None


def test_tavily_search_never_raises():
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("network down")

    with patch.object(search.httpx, "AsyncClient", Boom):
        assert run(search.tavily_search("q")) == []


# ---------------------------------------------------------------------------
# Gap-fill
# ---------------------------------------------------------------------------

def gap_reply(per_model):
    def reply(model, prompt):
        value = per_model.get(model)
        if value is None:
            return None
        return {"content": value}
    return reply


def test_gap_fill_ids_stable_and_cross_model_dupes_merged():
    m = COUNCIL_MODELS
    per_model = {
        m[0]: '{"queries": ["battery recycling rates 2026"]}',
        m[1]: 'Sure: {"queries": ["Battery  Recycling Rates 2026", "lithium price trend"]}',
        m[2]: '{"queries": []}',
    }
    rec = Recorder(gap_reply(per_model))
    base = ok_search(3)

    async def fake_tavily(q):
        slug = q.replace(" ", "-")
        return [result(f"https://{slug}.com/{i}", 0.9 - i * 0.1) for i in range(4)] + [
            result("https://site1.com/page/", 0.95)  # duplicate of existing source 1
        ]

    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "tavily_search", side_effect=fake_tavily):
        updated, added = run(council.stage0_5_gap_fill("question", base))

    queries = [q["query"] for q in added["queries"]]
    assert queries == ["battery recycling rates 2026", "lithium price trend"]
    assert added["queries"][0]["requested_by"] == [m[0], m[1]]

    # Existing IDs untouched, new IDs continue the numbering
    assert updated["sources"][:3] == base["sources"]
    assert [s["id"] for s in updated["sources"]] == [1, 2, 3, 4, 5, 6, 7]
    # Per-query cap, and the duplicate of source 1 was not re-added
    assert len(added["sources"]) == 4
    assert all("site1.com" not in s["url"] for s in added["sources"])
    assert "[7]" in updated["context"]


def test_gap_fill_prompt_identical_and_has_no_attribution():
    rec = Recorder('{"queries": []}')
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "tavily_search", AsyncMock(return_value=[])):
        run(council.stage0_5_gap_fill("question", ok_search(2)))
    prompts = set(rec.calls[0].values())
    assert len(prompts) == 1
    prompt = prompts.pop()
    for model in COUNCIL_MODELS:
        assert model not in prompt


def test_gap_fill_bad_json_from_one_model_is_ignored():
    m = COUNCIL_MODELS
    per_model = {
        m[0]: "I think you should search for {not json",
        m[1]: '{"queries": ["solar capacity india"]}',
        m[2]: None,  # model failed entirely
        m[3]: '{"queries": "not a list"}',
    }
    rec = Recorder(gap_reply(per_model))
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "tavily_search", AsyncMock(return_value=[result("https://new.com/a")])):
        updated, added = run(council.stage0_5_gap_fill("question", ok_search(2)))
    assert [q["query"] for q in added["queries"]] == ["solar capacity india"]
    assert updated["sources"][-1]["id"] == 3


def test_merge_gap_queries_caps_and_round_robin():
    requests = {
        "m1": ["alpha one", "alpha two", "alpha three"],  # third exceeds per-model cap
        "m2": ["beta one", "beta two"],
        "m3": ["gamma one", "gamma two"],
        "m4": ["delta one", "delta two"],
    }
    merged = council.merge_gap_queries(requests, ["original question"])
    queries = [q["query"] for q in merged]
    assert len(queries) == 6  # GAP_MAX_TOTAL_QUERIES
    assert "alpha three" not in queries
    # Round-robin: every model's first request comes before any second request
    assert queries[:4] == ["alpha one", "beta one", "gamma one", "delta one"]


def test_merge_gap_queries_drops_stage0_and_near_duplicates():
    requests = {
        "m1": ["Original  Question", "new zealand housing prices 2026 forecast"],
        "m2": ["housing prices new zealand 2026 forecast data"],  # Jaccard 6/7 >= 0.7
    }
    merged = council.merge_gap_queries(requests, ["original question"])
    # Round 1: m1's first request repeats Stage 0 and is dropped; m2's is accepted.
    # Round 2: m1's second request is a near-duplicate and only adds attribution.
    assert [q["query"] for q in merged] == ["housing prices new zealand 2026 forecast data"]
    assert merged[0]["requested_by"] == ["m2", "m1"]


def test_gap_fill_respects_pool_cap():
    m = COUNCIL_MODELS
    per_model = {model: f'{{"queries": ["topic {i} a", "topic {i} b"]}}' for i, model in enumerate(m)}
    rec = Recorder(gap_reply(per_model))

    async def fake_tavily(q):
        slug = q.replace(" ", "-")
        return [result(f"https://{slug}.com/{i}") for i in range(3)]

    base = ok_search(8)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "tavily_search", side_effect=fake_tavily):
        updated, added = run(council.stage0_5_gap_fill("question", base))
    assert len(updated["sources"]) == 12  # SEARCH_MAX_SOURCES_TOTAL
    assert [s["id"] for s in updated["sources"]] == list(range(1, 13))


def test_gap_fill_skipped_unless_status_ok():
    rec = Recorder()
    with patch.object(council, "query_models_parallel", rec):
        for status in ("failed", "skipped", "no_key"):
            s = {"status": status, "queries": [], "gap_queries": [], "sources": [], "context": ""}
            out, added = run(council.stage0_5_gap_fill("q", s))
            assert out is s and added == {"queries": [], "sources": []}
        out, _ = run(council.stage0_5_gap_fill("q", None))
        assert out is None
    assert rec.calls == []


# ---------------------------------------------------------------------------
# Prompts: identical evidence, no attribution, original prompts when off
# ---------------------------------------------------------------------------

def run_council_with(search_result, web_search=True):
    rec = Recorder(stage_reply)
    chair = AsyncMock(return_value={"content": "final [1]"})
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", chair), \
         patch.object(council, "run_stage0_search", AsyncMock(return_value=search_result)), \
         patch.object(council, "GAP_FILL_ENABLED", False):
        out = run(council.run_full_council("What is new?", web_search=web_search))
    return out, rec, chair


def test_all_stages_get_identical_evidence_block():
    s = ok_search(3)
    s["gap_queries"] = [{"query": "extra", "requested_by": [COUNCIL_MODELS[0]]}]
    (_, _, _, metadata), rec, chair = run_council_with(s)
    block = search.evidence_block(s["context"])

    stage1_prompts, stage2_prompts = rec.calls
    # Stage 1 prompt byte-identical across models
    assert len(set(stage1_prompts.values())) == 1
    assert len(set(stage2_prompts.values())) == 1
    stage3_prompt = chair.call_args.args[1][0]["content"]
    for prompt in (next(iter(stage1_prompts.values())), next(iter(stage2_prompts.values())), stage3_prompt):
        assert block in prompt

    # Stage 1 and 2 prompts reveal no model identity
    for prompt in (next(iter(stage1_prompts.values())), next(iter(stage2_prompts.values()))):
        for model in COUNCIL_MODELS:
            assert model not in prompt
    assert "requested_by" not in stage3_prompt
    assert metadata["search"]["status"] == "ok"
    assert all("content" not in src for src in metadata["search"]["sources"])


ORIGINAL_STAGE2_TASK = "1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.\n2. Then"
ORIGINAL_STAGE3_START = "You are the Chairman of an LLM Council."
ORIGINAL_STAGE3_LIST = "- Any patterns of agreement or disagreement\n\nProvide a clear"


@pytest.mark.parametrize("web_search, search_result", [
    (False, None),
    (True, {"status": "failed", "queries": ["q"], "gap_queries": [], "sources": [], "context": ""}),
    (True, {"status": "skipped", "queries": [], "gap_queries": [], "sources": [], "context": ""}),
    (True, {"status": "no_key", "queries": [], "gap_queries": [], "sources": [], "context": ""}),
])
def test_prompts_are_original_when_search_off_or_not_ok(web_search, search_result):
    (s1, s2, s3, metadata), rec, chair = run_council_with(search_result, web_search)
    stage1_prompts, stage2_prompts = rec.calls
    assert set(stage1_prompts.values()) == {"What is new?"}
    stage2_prompt = next(iter(stage2_prompts.values()))
    assert stage2_prompt.startswith("You are evaluating different responses")
    assert ORIGINAL_STAGE2_TASK in stage2_prompt
    stage3_prompt = chair.call_args.args[1][0]["content"]
    assert stage3_prompt.startswith(ORIGINAL_STAGE3_START)
    assert ORIGINAL_STAGE3_LIST in stage3_prompt
    assert "WEB SEARCH RESULTS" not in stage2_prompt + stage3_prompt
    # Council still completes
    assert len(s1) == len(COUNCIL_MODELS) and s3["response"] == "final [1]"
    if web_search is False:
        assert metadata["search"] is None


def test_prompts_without_context_match_original_code_exactly():
    """With no context, prompts must equal what the pre-search code produced."""
    stage1 = [{"model": "m1", "response": "r1"}, {"model": "m2", "response": "r2"}]
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
# SSE endpoint
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import main

    monkeypatch.setattr(storage, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "generate_conversation_title", AsyncMock(return_value="Title"))
    return TestClient(main.app)


def stream_events(client, body, search_result):
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", AsyncMock(return_value={"content": "final"})), \
         patch.object(council, "run_stage0_search", AsyncMock(return_value=search_result)) as s0, \
         patch.object(council, "tavily_search", AsyncMock(return_value=[result("https://gap.com/1")])):
        conv = client.post("/api/conversations", json={}).json()
        resp = client.post(f"/api/conversations/{conv['id']}/message/stream", json=body)
        events = [
            json.loads(line[6:]) for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        saved = client.get(f"/api/conversations/{conv['id']}").json()
    return events, saved, s0


def test_sse_event_order_search_on(client):
    events, saved, _ = stream_events(client, {"content": "Q?", "web_search": True}, ok_search(2))
    types = [e["type"] for e in events]
    assert types == [
        "search_start", "search_complete", "gap_start", "gap_complete",
        "stage1_start", "stage1_complete", "stage2_start", "stage2_complete",
        "stage3_start", "stage3_complete", "title_complete", "complete",
    ]
    search_complete = events[1]["data"]
    assert search_complete["status"] == "ok" and len(search_complete["sources"]) == 2
    assert "context" not in search_complete

    msg = saved["messages"][-1]
    assert msg["search"]["status"] == "ok"
    assert "context" not in msg["search"]


def test_sse_event_order_search_off(client):
    events, saved, s0 = stream_events(client, {"content": "Q?", "web_search": False}, ok_search(2))
    types = [e["type"] for e in events]
    assert types == [
        "stage1_start", "stage1_complete", "stage2_start", "stage2_complete",
        "stage3_start", "stage3_complete", "title_complete", "complete",
    ]
    s0.assert_not_called()
    assert "search" not in saved["messages"][-1]


def test_sse_search_failed_still_completes(client):
    failed = {"status": "failed", "queries": ["Q?"], "gap_queries": [], "sources": [], "context": ""}
    events, _, _ = stream_events(client, {"content": "Q?", "web_search": True}, failed)
    types = [e["type"] for e in events]
    assert types[:3] == ["search_start", "search_complete", "stage1_start"]
    assert events[1]["data"]["status"] == "failed"
    assert types[-1] == "complete"


def test_non_streaming_endpoint_persists_search(client):
    rec = Recorder(stage_reply)
    with patch.object(council, "query_models_parallel", rec), \
         patch.object(council, "query_model", AsyncMock(return_value={"content": "final"})), \
         patch.object(council, "run_stage0_search", AsyncMock(return_value=ok_search(2))), \
         patch.object(council, "GAP_FILL_ENABLED", False):
        conv = client.post("/api/conversations", json={}).json()
        data = client.post(f"/api/conversations/{conv['id']}/message", json={"content": "Q?"}).json()
        saved = client.get(f"/api/conversations/{conv['id']}").json()
    assert data["metadata"]["search"]["status"] == "ok"
    assert saved["messages"][-1]["search"]["sources"][0]["id"] == 1


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
