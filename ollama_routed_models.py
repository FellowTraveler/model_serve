"""OpenAI-format /v1/models entries for the models the proxy routes to Ollama.

llama-swap's /v1/models lists only the models it serves; routes with `backend: ollama` in
custom_models.yaml (gemma4, Jev, ...) work through the proxy but would be missing from
the list. Only routes whose Ollama model is installed are listed.
"""


def installed_ollama_tags(tags_response: dict) -> set[str]:
    """Model names from Ollama's GET /api/tags reply."""
    return {model.get("name", "") for model in tags_response.get("models", [])} - {""}


def is_installed(ollama_model: str, installed: set[str]) -> bool:
    """Ollama reports untagged names as name:latest."""
    return ollama_model in installed or f"{ollama_model}:latest" in installed


def ollama_routed_model_entries(custom_models: dict, installed: set[str], prefix: str, created: int) -> list[dict]:
    """One /v1/models entry per installed `backend: ollama` route, named as callers use it."""
    entries = []
    for name, settings in custom_models.items():
        settings = settings or {}
        ollama_model = settings.get("ollama_model")
        if settings.get("backend") != "ollama" or not ollama_model:
            continue
        if not is_installed(ollama_model, installed):
            continue
        entries.append({
            "id": f"{prefix}{name}",
            "object": "model",
            "created": created,
            "owned_by": "ollama",
            "meta": {"model_serve": {"backend": "ollama", "ollama_model": ollama_model}},
        })
    return entries


def merge_model_lists(llama_swap_list: dict, extra_entries: list[dict]) -> dict:
    """llama-swap's list plus entries whose id it doesn't already have."""
    data = list(llama_swap_list.get("data", []))
    known_ids = {entry.get("id") for entry in data}
    data.extend(entry for entry in extra_entries if entry["id"] not in known_ids)
    return {**llama_swap_list, "data": data}
