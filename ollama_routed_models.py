"""OpenAI-format /v1/models entries for the models the proxy routes to Ollama.

llama-swap's /v1/models lists only the models it serves; routes with `backend: ollama` in
custom_models.yaml (gemma4, Jev, ...) work through the proxy but would be missing from
the list. Only routes whose Ollama model is installed are listed.
"""
from typing import Optional


def installed_ollama_models(tags_response: dict) -> dict[str, list]:
    """Installed model name -> its capabilities, from Ollama's GET /api/tags reply.

    Ollama reports capabilities per model, e.g. ["completion", "tools"] or, for a
    decision-only model like Jev, ["decision"]; an empty list if it gives none.
    """
    return {
        model["name"]: model.get("capabilities") or []
        for model in tags_response.get("models", [])
        if model.get("name")
    }


def installed_name(ollama_model: str, installed: dict[str, list]) -> Optional[str]:
    """The name Ollama lists the model under (it reports untagged names as name:latest)."""
    for name in (ollama_model, f"{ollama_model}:latest"):
        if name in installed:
            return name
    return None


def ollama_routed_model_entries(custom_models: dict, installed: dict[str, list], prefix: str, created: int) -> list[dict]:
    """One /v1/models entry per installed `backend: ollama` route, named as callers use it.

    meta.model_serve.capabilities carries Ollama's capabilities, so clients can tell
    which models can chat ("completion") from e.g. decision-only ones.
    """
    entries = []
    for name, settings in custom_models.items():
        settings = settings or {}
        ollama_model = settings.get("ollama_model")
        if settings.get("backend") != "ollama" or not ollama_model:
            continue
        listed_as = installed_name(ollama_model, installed)
        if listed_as is None:
            continue
        entries.append({
            "id": f"{prefix}{name}",
            "object": "model",
            "created": created,
            "owned_by": "ollama",
            "meta": {"model_serve": {
                "backend": "ollama",
                "ollama_model": ollama_model,
                "capabilities": installed[listed_as],
            }},
        })
    return entries


def merge_model_lists(llama_swap_list: dict, extra_entries: list[dict]) -> dict:
    """llama-swap's list plus entries whose id it doesn't already have."""
    data = list(llama_swap_list.get("data", []))
    known_ids = {entry.get("id") for entry in data}
    data.extend(entry for entry in extra_entries if entry["id"] not in known_ids)
    return {**llama_swap_list, "data": data}
