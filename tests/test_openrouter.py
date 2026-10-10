"""Tests for OpenRouter retries and error handling. Fully mocked: no network."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend import openrouter, search
from backend.config import OPENROUTER_MAX_RETRIES


def run(coro):
    return asyncio.run(coro)


OK_BODY = {"choices": [{"message": {"content": "hi", "tool_calls": None}}]}


def reply(status=200, body=None, headers=None):
    return httpx.Response(status, json=OK_BODY if body is None else body, headers=headers or {})


class FakeClient:
    """Stands in for httpx.AsyncClient; replays scripted responses or exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.payloads = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.payloads.append(json)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def query(script, **kwargs):
    client = FakeClient(script)
    sleep = AsyncMock()
    with patch.object(openrouter.httpx, "AsyncClient", client), \
         patch.object(openrouter.asyncio, "sleep", sleep):
        out = run(openrouter.query_model("m1", [{"role": "user", "content": "q"}], **kwargs))
    return out, client, sleep


def test_success_first_try():
    out, client, sleep = query([reply()])
    assert out["content"] == "hi"
    assert len(client.payloads) == 1
    sleep.assert_not_called()


def test_429_then_success_honors_retry_after():
    out, client, sleep = query([reply(429, {"error": {"code": 429, "message": "slow down"}}, {"Retry-After": "3"}), reply()])
    assert out["content"] == "hi"
    assert len(client.payloads) == 2
    sleep.assert_awaited_once_with(3.0)


def test_retry_after_is_capped():
    _, _, sleep = query([reply(429, {}, {"Retry-After": "999"}), reply()])
    assert sleep.await_args.args[0] == openrouter.OPENROUTER_MAX_RETRY_AFTER


def test_backoff_grows_without_retry_after():
    _, _, sleep = query([reply(503, {}), reply(502, {}), reply()])
    first, second = [c.args[0] for c in sleep.await_args_list]
    assert 2.0 <= first < 2.5 and 4.0 <= second < 4.5


def test_gives_up_after_max_retries():
    script = [reply(429, {})] * (OPENROUTER_MAX_RETRIES + 1)
    out, client, sleep = query(script)
    assert out is None
    assert len(client.payloads) == OPENROUTER_MAX_RETRIES + 1
    assert sleep.await_count == OPENROUTER_MAX_RETRIES


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404])
def test_non_transient_errors_are_not_retried(status):
    out, client, sleep = query([reply(status, {"error": {"code": status, "message": "no"}})])
    assert out is None
    assert len(client.payloads) == 1
    sleep.assert_not_called()


def test_error_inside_200_is_retried():
    error_body = {"error": {"code": 502, "message": "provider died"}}
    out, client, _ = query([reply(200, error_body), reply()])
    assert out["content"] == "hi"
    assert len(client.payloads) == 2


def test_connection_error_retried_but_read_timeout_not():
    out, client, _ = query([httpx.ConnectError("refused"), reply()])
    assert out["content"] == "hi" and len(client.payloads) == 2

    out, client, sleep = query([httpx.ReadTimeout("slow")])
    assert out is None and len(client.payloads) == 1
    sleep.assert_not_called()


def test_tools_require_tool_capable_provider():
    _, client, _ = query([reply()], tools=[search.WEB_SEARCH_TOOL], tool_choice="auto")
    payload = client.payloads[0]
    assert payload["provider"] == {"require_parameters": True}
    assert payload["tools"] == [search.WEB_SEARCH_TOOL] and payload["tool_choice"] == "auto"

    _, client, _ = query([reply()])
    assert "provider" not in client.payloads[0] and "tools" not in client.payloads[0]


def test_error_log_names_provider_and_type(capsys):
    body = {"error": {"code": 429, "message": "Rate limited", "metadata": {
        "error_type": "rate_limit_exceeded", "provider_name": "Google"}}}
    query([reply(429, body), reply()])
    out = capsys.readouterr().out
    assert "HTTP 429: Rate limited (error_type=rate_limit_exceeded, provider_name=Google)" in out
