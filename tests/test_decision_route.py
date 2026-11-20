"""Tests for the decision-model pass-through (POST /v1/systemone -> Ollama)."""
import json
import logging

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import decision_route
from decision_route import DECISION_PATH, build_decision_body, create_decision_router

OLLAMA_BASE = "http://ollama.test"
QUESTIONS = {"bug": {"type": "noul", "instructions": "Is this a bug report?",
                     "criteria": {"true": "broken behaviour", "false": "anything else"}}}


def test_body_maps_model_and_adds_keep_alive_as_duration_string():
    body = build_decision_body({"model": "ms/jev", "state": "s", "questions": QUESTIONS},
                               "autotrust-jev:9b", 1800)
    assert body["model"] == "autotrust-jev:9b"
    assert body["keep_alive"] == "1800s"  # the endpoint ignores a numeric keep_alive
    assert body["questions"] == QUESTIONS


def test_body_keeps_the_callers_keep_alive():
    body = build_decision_body({"model": "m", "keep_alive": "5m"}, "m", 1800)
    assert body["keep_alive"] == "5m"


def make_client(monkeypatch, handler):
    """App with the decision router, whose outgoing httpx calls go to `handler`."""
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(decision_route.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(handler)))
    app = FastAPI()
    app.include_router(create_decision_router(
        ollama_base=OLLAMA_BASE,
        keep_alive_seconds=1800,
        map_model_name=lambda name: {"ms/jev": "autotrust-jev:9b"}.get(name, name),
        to_response=lambda resp: decision_route.JSONResponse(resp.json(), resp.status_code),
        logger=logging.getLogger("test"),
    ))
    return TestClient(app)


def test_route_forwards_to_ollama_and_returns_its_answer(monkeypatch):
    seen = {}

    def ollama(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {"bug": {"type": "noul", "noul": 0.91}}})

    client = make_client(monkeypatch, ollama)
    resp = client.post(DECISION_PATH, json={"model": "ms/jev", "state": "crash on start", "questions": QUESTIONS})
    assert resp.status_code == 200
    assert resp.json()["answers"]["bug"]["noul"] == 0.91
    assert seen["url"] == f"{OLLAMA_BASE}{DECISION_PATH}"
    assert seen["body"]["model"] == "autotrust-jev:9b"
    assert seen["body"]["keep_alive"] == "1800s"


def test_route_passes_ollama_errors_through(monkeypatch):
    client = make_client(monkeypatch, lambda request: httpx.Response(404, json={"error": "model 'x' not found"}))
    resp = client.post(DECISION_PATH, json={"model": "x", "state": "s", "questions": {}})
    assert resp.status_code == 404
    assert "not found" in resp.json()["error"]


def test_route_reports_unreachable_ollama_as_502(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    client = make_client(monkeypatch, refuse)
    resp = client.post(DECISION_PATH, json={"model": "x", "state": "s", "questions": {}})
    assert resp.status_code == 502


def test_proxy_app_serves_the_route_and_maps_jev(monkeypatch):
    import harmony_proxy
    seen = {}

    def ollama(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"answers": {}})

    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(decision_route.httpx, "AsyncClient",
                        lambda **kwargs: real_async_client(transport=httpx.MockTransport(ollama)))
    resp = TestClient(harmony_proxy.app).post(
        DECISION_PATH, json={"model": "ms/autotrust-jev-9b", "state": "s", "questions": {}})
    assert resp.status_code == 200
    assert seen["url"] == f"{harmony_proxy.OLLAMA_BASE}{DECISION_PATH}"
    assert seen["body"]["model"] == "autotrust-jev:9b"
    assert seen["body"]["keep_alive"] == f"{harmony_proxy.OLLAMA_KEEP_ALIVE_SECONDS}s"
