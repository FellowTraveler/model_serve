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


@dataclass
class LoadedModel:
    """Represents a loaded model with its metadata."""
    name: str
    backend: str  # "llama-swap" or "ollama"
    last_seen: float  # timestamp when last seen in running list
    size_bytes: int = 0  # VRAM size if known


class ModelTracker:
    """Tracks loaded models and their last-used times across backends."""

    def __init__(self):
        # model_name -> LoadedModel
        self.models: dict[str, LoadedModel] = {}

    def update_from_llama_swap(self, running_models: list[str]):
        """Update tracker with models from llama-swap."""
        now = time.time()
        seen = set()

        for name in running_models:
            seen.add(name)
            if name in self.models:
                # Model still running - update last_seen
                self.models[name].last_seen = now
            else:
                # New model appeared
                self.models[name] = LoadedModel(
                    name=name,
                    backend="llama-swap",
                    last_seen=now,
                )
                logger.info(f"Tracking new llama-swap model: {name}")

        # Remove models no longer running in llama-swap
        for name in list(self.models.keys()):
            if self.models[name].backend == "llama-swap" and name not in seen:
                logger.info(f"Model no longer running: {name}")
                del self.models[name]

    def update_from_ollama(self, ollama_models: list[dict]):
        """Update tracker with models from Ollama /api/ps."""
        now = time.time()
        seen = set()

        for m in ollama_models:
            name = m.get("name", "")
            if not name:
                continue

            seen.add(name)
            size = m.get("size_vram", 0)

            if name in self.models:
                self.models[name].last_seen = now
                self.models[name].size_bytes = size
            else:
                self.models[name] = LoadedModel(
                    name=name,
                    backend="ollama",
                    last_seen=now,
                    size_bytes=size,
                )
                logger.info(f"Tracking new Ollama model: {name}")

        # Remove models no longer running in Ollama
        for name in list(self.models.keys()):
            if self.models[name].backend == "ollama" and name not in seen:
                logger.info(f"Model no longer running: {name}")
                del self.models[name]

    def get_lru_candidate(self, min_age_seconds: float = 5.0) -> Optional[LoadedModel]:
        """
        Get the least recently used model that's old enough to unload.

        Args:
            min_age_seconds: Don't unload models used more recently than this
        """
        now = time.time()
        candidates = [
            m for m in self.models.values()
            if (now - m.last_seen) >= min_age_seconds
        ]

        if not candidates:
            return None

        # Sort by last_seen ascending (oldest first)
        candidates.sort(key=lambda m: m.last_seen)
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


def get_llama_swap_models(api_url: str) -> list[str]:
    """Fetch list of currently loaded models from llama-swap."""
    try:
        response = requests.get(f"{api_url}/running", timeout=5)
        response.raise_for_status()
        data = response.json()
        # llama-swap returns {"running": ["model1", "model2", ...]}
        return data.get('running', []) if isinstance(data, dict) else data
    except requests.exceptions.RequestException as e:
        logger.debug(f"Failed to get llama-swap models: {e}")
        return []


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
                        age = time.time() - candidate.last_seen
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
