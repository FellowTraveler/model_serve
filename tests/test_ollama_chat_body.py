"""Tests for the OpenAI -> Ollama /api/chat request translation."""
from harmony_proxy import (
    OLLAMA_KEEP_ALIVE_SECONDS,
    build_ollama_chat_body,
    get_model_ctx_size,
)

OLLAMA_ROUTED_MODEL = "ms/gemma4-26b-a4b-it-q4_k_m"


def test_keeps_model_loaded_for_model_ttl():
    """Ollama unloads after 5 minutes unless told otherwise; the proxy asks for MODEL_TTL."""
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": []}, stream=False)
    assert body["keep_alive"] == OLLAMA_KEEP_ALIVE_SECONDS


def test_uses_configured_ollama_name_and_context():
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": []}, stream=False)
    assert body["model"] == "gemma4:26b-a4b-it-q4_K_M"
    assert body["options"]["num_ctx"] == get_model_ctx_size(OLLAMA_ROUTED_MODEL)


def test_request_overrides_win():
    body = build_ollama_chat_body(
        {"model": OLLAMA_ROUTED_MODEL, "messages": [], "temperature": 0, "max_tokens": 7},
        stream=True,
    )
    assert body["options"]["temperature"] == 0
    assert body["options"]["num_predict"] == 7
    assert body["stream"] is True


def test_json_object_response_format_becomes_ollama_json():
    body = build_ollama_chat_body(
        {"model": OLLAMA_ROUTED_MODEL, "messages": [], "response_format": {"type": "json_object"}},
        stream=False,
    )
    assert body["format"] == "json"


def test_json_schema_response_format_passes_the_schema():
    schema = {"type": "object", "properties": {"bucket": {"type": "string"}}}
    body = build_ollama_chat_body(
        {
            "model": OLLAMA_ROUTED_MODEL,
            "messages": [],
            "response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": schema}},
        },
        stream=False,
    )
    assert body["format"] == schema


def test_no_response_format_means_free_text():
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": []}, stream=False)
    assert "format" not in body


def test_without_sampling_options_the_route_settings_apply():
    """RustyMail's model_serve provider sends no sampling; the route's sampler_args decide."""
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": []}, stream=False)
    assert body["options"]["temperature"] == 1.0
    assert body["options"]["top_p"] == 0.95
    assert body["options"]["top_k"] == 64
    assert body["think"] is False


def test_tools_are_passed_through():
    tools = [{"type": "function", "function": {"name": "search_email", "parameters": {"type": "object"}}}]
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": [], "tools": tools}, stream=False)
    assert body["tools"] == tools


def test_explicit_zero_values_override_the_route():
    """temperature 0 (deterministic triage) and seed 0 are real settings, not 'unset'."""
    body = build_ollama_chat_body(
        {"model": OLLAMA_ROUTED_MODEL, "messages": [], "temperature": 0, "top_p": 0.0, "seed": 0},
        stream=False,
    )
    assert body["options"]["temperature"] == 0
    assert body["options"]["top_p"] == 0.0
    assert body["options"]["seed"] == 0


def test_all_request_sampling_fields_are_forwarded():
    sent = {"top_k": 1, "min_p": 0.1, "stop": ["\n\n"], "presence_penalty": 0.5,
            "frequency_penalty": 0.25, "repeat_penalty": 1.1, "max_completion_tokens": 9}
    body = build_ollama_chat_body({"model": OLLAMA_ROUTED_MODEL, "messages": [], **sent}, stream=False)
    for key in ("top_k", "min_p", "stop", "presence_penalty", "frequency_penalty", "repeat_penalty"):
        assert body["options"][key] == sent[key]
    assert body["options"]["num_predict"] == 9


def test_log_reports_the_options_actually_sent(monkeypatch, caplog):
    import json
    import logging

    import httpx
    from fastapi.testclient import TestClient

    import harmony_proxy

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(harmony_proxy.httpx, "AsyncClient", lambda **kwargs: real_async_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"message": {"role": "assistant", "content": "{}"}, "done": True}))))
    with caplog.at_level(logging.INFO, logger="harmony_proxy"):
        TestClient(harmony_proxy.app).post("/v1/chat/completions", json={
            "model": OLLAMA_ROUTED_MODEL, "temperature": 0,
            "response_format": {"type": "json_object"}, "messages": [{"role": "user", "content": "x"}],
        })
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Ollama chat:"))
    assert "'temperature': 0," in line
