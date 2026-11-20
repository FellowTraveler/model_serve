#!/usr/bin/env python3
"""
Pressure-aware model unloader for llama-swap and Ollama.

Monitors system memory and unloads idle models when memory pressure is high.
Uses LRU (Least Recently Used) strategy to pick which model to unload.

Supports:
- llama-swap: /running endpoint, /unload endpoint
- Ollama: /api/ps endpoint, keep_alive:0 for unloading
"""

import os
import sys
import time
import logging
from typing import Optional
from dataclasses import dataclass
import requests
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# llama-server --metrics counters that only grow when the model serves a request.
LLAMA_SERVER_ACTIVITY_METRICS = ("llamacpp:prompt_tokens_total", "llamacpp:tokens_predicted_total")


@dataclass
class LoadedModel:
    """Represents a loaded model with its metadata."""
    name: str
    backend: str  # "llama-swap" or "ollama"
    last_used: float  # when the model last served a request (or was first seen)
    size_bytes: int = 0  # VRAM size if known
    activity: object = None  # backend usage marker; a change means the model was used


class ModelTracker:
    """Tracks loaded models and their last-used times across backends."""

    def __init__(self):
        # model_name -> LoadedModel
        self.models: dict[str, LoadedModel] = {}

    def observe(self, name: str, backend: str, activity, size_bytes: int = 0, now: Optional[float] = None):
        """Record a loaded model; it counts as used when its activity marker changes.

        A missing marker (None) counts as use, so a model we can't see into is never
        mistaken for idle.
        """
        now = time.time() if now is None else now
        model = self.models.get(name)
        if model is None:
            self.models[name] = LoadedModel(name, backend, now, size_bytes, activity)
            logger.info(f"Tracking new {backend} model: {name}")
            return
        if activity is None or activity != model.activity:
            model.last_used = now
            model.activity = activity
        if size_bytes:
            model.size_bytes = size_bytes

    def forget_missing(self, backend: str, running_names: set[str]):
        """Drop models of this backend that are no longer loaded."""
        for name in list(self.models.keys()):
            if self.models[name].backend == backend and name not in running_names:
                logger.info(f"Model no longer running: {name}")
                del self.models[name]

    def update_from_llama_swap(self, running_models: list[tuple[str, object]]):
        """Update tracker with (name, activity) pairs from llama-swap."""
        for name, activity in running_models:
            self.observe(name, "llama-swap", activity)
        self.forget_missing("llama-swap", {name for name, _ in running_models})

    def update_from_ollama(self, ollama_models: list[dict]):
        """Update tracker with models from Ollama /api/ps; expires_at moves on every request."""
        running_names = set()
        for m in ollama_models:
            name = m.get("name", "")
            if not name:
                continue
            running_names.add(name)
            self.observe(name, "ollama", m.get("expires_at"), m.get("size_vram", 0))
        self.forget_missing("ollama", running_names)

    def get_lru_candidate(self, min_age_seconds: float = 5.0) -> Optional[LoadedModel]:
        """
        Get the least recently used model that's old enough to unload.

        Args:
            min_age_seconds: Don't unload models used more recently than this
        """
        now = time.time()
        candidates = [
            m for m in self.models.values()
            if (now - m.last_used) >= min_age_seconds
        ]

        if not candidates:
            return None

        # Sort by last_used ascending (least recently used first)
        candidates.sort(key=lambda m: m.last_used)
        return candidates[0]


def get_config():
    """Load configuration from environment variables."""
    llama_swap_port = os.environ.get('LLAMA_SWAP_PORT', '5847')
    return {
        'llama_swap_url': f"http://127.0.0.1:{llama_swap_port}",
        'ollama_url': os.environ.get('OLLAMA_BASE', 'http://localhost:11434'),
        'memory_threshold': int(os.environ.get('MEMORY_PRESSURE_THRESHOLD', '75')),
        'check_interval': int(os.environ.get('PRESSURE_CHECK_INTERVAL', '30')),
        'min_model_age': float(os.environ.get('MIN_MODEL_AGE_SECONDS', '5')),
    }


def get_llama_server_activity(proxy_url: str) -> Optional[tuple]:
    """Read a llama-server's request counters from /metrics; None if unreadable."""
    try:
        response = requests.get(f"{proxy_url}/metrics", timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to read metrics from {proxy_url}: {e}")
        return None
    values = {}
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in LLAMA_SERVER_ACTIVITY_METRICS:
            values[parts[0]] = parts[1]
    if len(values) != len(LLAMA_SERVER_ACTIVITY_METRICS):
        return None
    return tuple(values[name] for name in LLAMA_SERVER_ACTIVITY_METRICS)


def get_llama_swap_models(api_url: str) -> list[tuple[str, object]]:
    """Fetch (name, activity) for each model llama-swap has loaded."""
    try:
        response = requests.get(f"{api_url}/running", timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to get llama-swap models: {e}")
        return []
    # llama-swap returns {"running": [{"model": ..., "proxy": ..., "state": ...}, ...]}
    running = []
    for entry in data.get('running', []) if isinstance(data, dict) else data:
        if isinstance(entry, dict) and entry.get("model"):
            proxy_url = entry.get("proxy")
            activity = get_llama_server_activity(proxy_url) if proxy_url else None
            running.append((entry["model"], activity))
    return running


def get_ollama_models(api_url: str) -> list[dict]:
    """Fetch list of currently loaded models from Ollama."""
    try:
        response = requests.get(f"{api_url}/api/ps", timeout=5)
        response.raise_for_status()
        data = response.json()
        return data.get('models', [])
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to get Ollama models: {e}")
        return []


def unload_llama_swap_model(api_url: str, model_name: str) -> bool:
    """Request llama-swap to unload a specific model."""
    try:
        response = requests.post(
            f"{api_url}/unload",
            params={"model": model_name},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Successfully unloaded llama-swap model: {model_name}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to unload llama-swap model {model_name}: {e}")
        return False


def unload_ollama_model(api_url: str, model_name: str) -> bool:
    """Request Ollama to unload a specific model using keep_alive: 0."""
    try:
        response = requests.post(
            f"{api_url}/api/generate",
            json={
                "model": model_name,
                "prompt": "",
                "keep_alive": 0,
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        if data.get("done_reason") == "unload":
            logger.info(f"Successfully unloaded Ollama model: {model_name}")
            return True
        else:
            logger.warning(f"Ollama unload response unexpected: {data}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to unload Ollama model {model_name}: {e}")
        return False


def unload_model(model: LoadedModel, config: dict) -> bool:
    """Unload a model using the appropriate backend."""
    if model.backend == "llama-swap":
        return unload_llama_swap_model(config['llama_swap_url'], model.name)
    elif model.backend == "ollama":
        return unload_ollama_model(config['ollama_url'], model.name)
    else:
        logger.error(f"Unknown backend: {model.backend}")
        return False


def get_memory_usage() -> float:
    """Get current system memory usage percentage."""
    return psutil.virtual_memory().percent


def main():
    """Main monitoring loop."""
    config = get_config()
    tracker = ModelTracker()

    logger.info("Starting pressure-aware model unloader")
    logger.info(f"llama-swap URL: {config['llama_swap_url']}")
    logger.info(f"Ollama URL: {config['ollama_url']}")
    logger.info(f"Memory threshold: {config['memory_threshold']}%")
    logger.info(f"Check interval: {config['check_interval']}s")
    logger.info(f"Min model age before unload: {config['min_model_age']}s")

    while True:
        try:
            # Update tracker with current state from both backends
            llama_swap_models = get_llama_swap_models(config['llama_swap_url'])
            ollama_models = get_ollama_models(config['ollama_url'])

            tracker.update_from_llama_swap(llama_swap_models)
            tracker.update_from_ollama(ollama_models)

            memory_usage = get_memory_usage()
            total_models = len(tracker.models)

            if memory_usage > config['memory_threshold']:
                logger.warning(
                    f"Memory pressure detected: {memory_usage:.1f}% "
                    f"(threshold: {config['memory_threshold']}%, "
                    f"models loaded: {total_models})"
                )

                if total_models > 0:
                    # Get LRU candidate (oldest model that's been idle long enough)
                    candidate = tracker.get_lru_candidate(config['min_model_age'])

                    if candidate:
                        age = time.time() - candidate.last_used
                        logger.info(
                            f"Unloading LRU candidate: {candidate.name} "
                            f"(backend: {candidate.backend}, idle: {age:.1f}s)"
                        )
                        if unload_model(candidate, config):
                            # Remove from tracker immediately
                            del tracker.models[candidate.name]
                    else:
                        logger.info(
                            f"No models idle long enough to unload "
                            f"(min age: {config['min_model_age']}s)"
                        )
                else:
                    logger.info("No models currently loaded")
            else:
                logger.debug(
                    f"Memory OK: {memory_usage:.1f}% "
                    f"(models: {total_models})"
                )

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

        time.sleep(config['check_interval'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down pressure unloader")
        sys.exit(0)
