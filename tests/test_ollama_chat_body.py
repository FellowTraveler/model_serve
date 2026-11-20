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
