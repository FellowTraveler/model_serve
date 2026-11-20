"""Pass-through for Ollama's decision endpoint, for models with the `decision` capability (e.g. Jev).

A decision model takes a text `state` and named yes/no ("noul") or multiple-choice ("choice")
`questions` and returns probabilities instead of generated text. Routing it through the proxy
gives callers one endpoint for every local model, maps `ms/` names to Ollama names, and keeps
the model loaded for MODEL_TTL like the other Ollama routes.
"""
from typing import Callable

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# Ollama's decision API path; served at the same path on the proxy.
DECISION_PATH = "/v1/systemone"


def build_decision_body(body: dict, ollama_model: str, keep_alive_seconds: int) -> dict:
    """The request to forward: Ollama's model name, and a keep_alive unless the caller set one.

    This endpoint ignores a numeric keep_alive (measured on Ollama 0.40.16) and honours a
    duration string, so it is sent as e.g. "1800s".
    """
    forwarded = {**body, "model": ollama_model}
    forwarded.setdefault("keep_alive", f"{keep_alive_seconds}s")
    return forwarded


def create_decision_router(
    ollama_base: str,
    keep_alive_seconds: int,
    map_model_name: Callable[[str], str],
    to_response: Callable[[httpx.Response], JSONResponse],
    logger,
) -> APIRouter:
    """Router forwarding POST /v1/systemone to Ollama, with the proxy's naming and keep-alive."""
    router = APIRouter()

    @router.post(DECISION_PATH)
    async def decide(request: Request):
        body = await request.json()
        model = body.get("model", "")
        ollama_model = map_model_name(model)
        logger.info(f"Decision request: {model} -> {ollama_model} via Ollama")
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                resp = await client.post(
                    f"{ollama_base}{DECISION_PATH}",
                    json=build_decision_body(body, ollama_model, keep_alive_seconds),
                )
                return to_response(resp)
            except httpx.HTTPError as e:
                logger.error(f"Ollama decision request failed: {e}")
                return JSONResponse(
                    content={"error": {"message": str(e), "type": "server_error"}},
                    status_code=502,
                )

    return router
