"""Conversations that include a tool call and its result, on Ollama routes."""
import copy
import json

import httpx
from fastapi.testclient import TestClient

import harmony_proxy
from harmony_proxy import ollama_messages

OLLAMA_ROUTED_MODEL = "ms/gemma4-26b-a4b-it-q4_k_m"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
MESSAGES = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\": 14, \"sky\": \"rain\"}"},
]
ANSWER = "The weather in Paris is 14°C with rain."


def test_string_arguments_become_objects_without_touching_the_input():
    original = copy.deepcopy(MESSAGES)
    converted = ollama_messages(MESSAGES)
    assert converted[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert converted[1]["tool_calls"][0]["id"] == "call_1"
    assert converted[0] == MESSAGES[0] and converted[2] == MESSAGES[2]  # user and tool unchanged
    assert MESSAGES == original


def test_object_empty_and_invalid_arguments():
    def args_after(arguments):
        message = {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": arguments}}]}
        return ollama_messages([message])[0]["tool_calls"][0]["function"]["arguments"]

    assert args_after({"city": "Paris"}) == {"city": "Paris"}
    assert args_after("") == {}
    assert args_after("{not json") == "{not json"  # passed on; Ollama's error reaches the client


def fake_ollama(monkeypatch, stream: bool, reject_with=None):
    """Behaves like Ollama /api/chat: 400 if any tool-call arguments is a string."""
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        if reject_with:
            return httpx.Response(400, json={"error": reject_with})
        for message in body["messages"]:
            for call in message.get("tool_calls") or []:
                if isinstance(call["function"]["arguments"], str):
                    return httpx.Response(400, json={"error": "Value looks like object, but can't find closing '}' symbol"})
        if stream:
            lines = [{"message": {"content": ANSWER}, "done": False}, {"message": {"content": ""}, "done": True}]
            return httpx.Response(200, content="".join(json.dumps(l) + "\n" for l in lines).encode())
        return httpx.Response(200, json={"message": {"role": "assistant", "content": ANSWER}, "done": True,
                                         "prompt_eval_count": 40, "eval_count": 12})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(harmony_proxy.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler)))


def post(stream: bool):
    return TestClient(harmony_proxy.app).post("/v1/chat/completions", json={
        "model": OLLAMA_ROUTED_MODEL, "stream": stream, "tools": TOOLS, "messages": MESSAGES})


def test_tool_result_conversation_gets_an_answer(monkeypatch):
    fake_ollama(monkeypatch, stream=False)
    resp = post(stream=False)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == ANSWER
    assert resp.json()["usage"]["completion_tokens"] == 12


def test_tool_result_conversation_streams_an_answer(monkeypatch):
    fake_ollama(monkeypatch, stream=True)
    resp = post(stream=True)
    assert resp.status_code == 200
    text = "".join(
        json.loads(line[len("data: "):])["choices"][0]["delta"].get("content", "")
        for line in resp.text.splitlines() if line.startswith("data: {"))
    assert text == ANSWER


def test_ollama_error_is_returned_not_an_empty_reply(monkeypatch):
    fake_ollama(monkeypatch, stream=False, reject_with="model 'x' not found")
    resp = post(stream=False)
    assert resp.status_code == 400
    assert "model 'x' not found" in resp.json()["error"]["message"]


def test_ollama_error_is_returned_when_streaming(monkeypatch):
    fake_ollama(monkeypatch, stream=True, reject_with="model 'x' not found")
    resp = post(stream=True)
    assert resp.status_code == 400
    assert "model 'x' not found" in resp.json()["error"]["message"]
