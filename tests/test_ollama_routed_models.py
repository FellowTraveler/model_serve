"""Tests for listing Ollama-routed models in GET /v1/models."""
import httpx
from fastapi.testclient import TestClient

import harmony_proxy
from ollama_routed_models import (
    installed_ollama_models,
    merge_model_lists,
    ollama_routed_model_entries,
)

CUSTOM = {
    "gemma4-26b-a4b-it-q4_k_m": {"backend": "ollama", "ollama_model": "gemma4:26b-a4b-it-q4_K_M"},
    "autotrust-jev-9b": {"backend": "ollama", "ollama_model": "autotrust-jev:9b"},
    "medgemma1.5-4.3b-q4_k_m": {"backend": "ollama", "ollama_model": "medgemma1.5"},
    "not-pulled": {"backend": "ollama", "ollama_model": "gone:1b"},
    "llama-swap-model": {"jinja": True},
    "lm-studio-model": {"backend": "lm_studio", "lm_studio_model": "x"},
    "empty-entry": None,
}
INSTALLED = {
    "gemma4:26b-a4b-it-q4_K_M": ["completion", "vision", "tools", "thinking"],
    "autotrust-jev:9b": ["decision"],
    "medgemma1.5:latest": ["completion"],
}


def test_lists_only_installed_ollama_routes_with_prefixed_names():
    entries = ollama_routed_model_entries(CUSTOM, INSTALLED, "ms/", 123)
    ids = [entry["id"] for entry in entries]
    assert ids == ["ms/gemma4-26b-a4b-it-q4_k_m", "ms/autotrust-jev-9b", "ms/medgemma1.5-4.3b-q4_k_m"]
    assert entries[0]["object"] == "model"
    assert entries[0]["owned_by"] == "ollama"
    assert entries[0]["meta"]["model_serve"]["ollama_model"] == "gemma4:26b-a4b-it-q4_K_M"


def test_entries_carry_ollama_capabilities():
    """Clients keep decision-only models like Jev out of chat menus by these."""
    entries = {e["id"]: e for e in ollama_routed_model_entries(CUSTOM, INSTALLED, "ms/", 123)}
    assert entries["ms/autotrust-jev-9b"]["meta"]["model_serve"]["capabilities"] == ["decision"]
    assert "completion" in entries["ms/gemma4-26b-a4b-it-q4_k_m"]["meta"]["model_serve"]["capabilities"]
    # an untagged ollama_model is found under Ollama's name:latest
    assert entries["ms/medgemma1.5-4.3b-q4_k_m"]["meta"]["model_serve"]["capabilities"] == ["completion"]


def test_installed_models_and_capabilities_come_from_api_tags():
    tags = {"models": [{"name": "a:1b", "capabilities": ["decision"]}, {"name": "b:latest"}]}
    assert installed_ollama_models(tags) == {"a:1b": ["decision"], "b:latest": []}


def test_merge_keeps_llama_swap_entries_and_skips_duplicate_ids():
    llama_swap = {"object": "list", "data": [{"id": "ms/a"}]}
    merged = merge_model_lists(llama_swap, [{"id": "ms/a"}, {"id": "ms/b"}])
    assert merged["object"] == "list"
    assert [entry["id"] for entry in merged["data"]] == ["ms/a", "ms/b"]


def fake_backends(monkeypatch, ollama_up: bool):
    def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "ms/qwen3-4.0b-q8_0", "object": "model"}]})
        if request.url.path == "/api/tags":
            if not ollama_up:
                raise httpx.ConnectError("refused")
            return httpx.Response(200, json={"models": [{"name": "autotrust-jev:9b", "capabilities": ["decision"]}]})
        return httpx.Response(404)

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(harmony_proxy.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler)))


def test_endpoint_adds_ollama_routes(monkeypatch):
    fake_backends(monkeypatch, ollama_up=True)
    models = {m["id"]: m for m in TestClient(harmony_proxy.app).get("/v1/models").json()["data"]}
    assert "ms/qwen3-4.0b-q8_0" in models
    assert models["ms/autotrust-jev-9b"]["meta"]["model_serve"]["capabilities"] == ["decision"]


def test_endpoint_still_lists_llama_swap_models_when_ollama_is_down(monkeypatch):
    fake_backends(monkeypatch, ollama_up=False)
    resp = TestClient(harmony_proxy.app).get("/v1/models")
    assert resp.status_code == 200
    assert [m["id"] for m in resp.json()["data"]] == ["ms/qwen3-4.0b-q8_0"]


def test_endpoint_reports_unreachable_llama_swap_as_502(monkeypatch):
    def refuse(request: httpx.Request):
        raise httpx.ConnectError("All connection attempts failed")

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(harmony_proxy.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(refuse)))
    resp = TestClient(harmony_proxy.app, raise_server_exceptions=False).get("/v1/models")
    assert resp.status_code == 502
    assert "llama-swap unavailable" in resp.json()["error"]["message"]
