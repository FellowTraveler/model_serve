"""Streaming through an Ollama route: content and tool calls become OpenAI SSE chunks."""
import json

import httpx
from fastapi.testclient import TestClient

import harmony_proxy

OLLAMA_ROUTED_MODEL = "ms/gemma4-26b-a4b-it-q4_k_m"
TOOLS = [{"type": "function", "function": {"name": "search_email", "parameters": {"type": "object"}}}]


def stream_through_proxy(monkeypatch, ollama_lines: list[dict]) -> list[dict]:
    """POST a streamed chat to the proxy with Ollama faked; return the parsed SSE chunks."""
    ndjson = "".join(json.dumps(line) + "\n" for line in ollama_lines).encode()
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(harmony_proxy.httpx, "AsyncClient", lambda **kwargs: real_async_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=ndjson))))
    resp = TestClient(harmony_proxy.app).post("/v1/chat/completions", json={
        "model": OLLAMA_ROUTED_MODEL, "stream": True, "tools": TOOLS,
        "messages": [{"role": "user", "content": "Find emails from Dana"}],
    })
    assert resp.status_code == 200
    events = [line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    return [json.loads(event) for event in events[:-1]]


def test_streamed_tool_call_reaches_the_client(monkeypatch):
    chunks = stream_through_proxy(monkeypatch, [
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"index": 0, "name": "search_email", "arguments": {"query": "from:Dana"}}},
        ]}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
    ])
    tool_deltas = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert tool_deltas == [[{
        "id": "call_1", "type": "function", "index": 0,
        "function": {"name": "search_email", "arguments": json.dumps({"query": "from:Dana"})},
    }]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_calls_in_separate_chunks_get_consecutive_indexes(monkeypatch):
    call = {"function": {"name": "search_email", "arguments": {}}}
    chunks = stream_through_proxy(monkeypatch, [
        {"message": {"content": "", "tool_calls": [call]}, "done": False},
        {"message": {"content": "", "tool_calls": [call]}, "done": False},
        {"message": {"content": ""}, "done": True},
    ])
    indexes = [tc["index"] for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls", [])]
    assert indexes == [0, 1]


def test_streamed_text_ends_with_stop(monkeypatch):
    chunks = stream_through_proxy(monkeypatch, [
        {"message": {"content": "Hel"}, "done": False},
        {"message": {"content": "lo"}, "done": True},
    ])
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hello"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
