#!/usr/bin/env python3
"""
Pressure-aware model unloader for llama-swap.

Monitors system memory and unloads idle models when memory pressure is high.
Works alongside llama-swap's TTL-based unloading.
"""

import os
import sys
import time
import logging
import requests
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_config():
    """Load configuration from environment variables."""
    port = os.environ.get('LLAMA_SWAP_PORT', '5847')
    return {
        'api_url': f"http://127.0.0.1:{port}",
        'memory_threshold': int(os.environ.get('MEMORY_PRESSURE_THRESHOLD', '75')),
        'check_interval': int(os.environ.get('PRESSURE_CHECK_INTERVAL', '30')),
    }


def get_running_models(api_url: str) -> list:
    """Fetch list of currently loaded models from llama-swap."""
    try:
        response = requests.get(f"{api_url}/running", timeout=5)
        response.raise_for_status()
        data = response.json()
        # llama-swap returns {"running": ["model1", "model2", ...]}
        return data.get('running', []) if isinstance(data, dict) else data
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get running models: {e}")
        return []


def unload_model(api_url: str, model_name: str) -> bool:
    """Request llama-swap to unload a specific model."""
    try:
        response = requests.post(
            f"{api_url}/models/unload",
            json={"model": model_name},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Successfully unloaded model: {model_name}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to unload model {model_name}: {e}")
        return False


def get_memory_usage() -> float:
    """Get current system memory usage percentage."""
    return psutil.virtual_memory().percent


def select_unload_candidate(models: list) -> str | None:
    """
    Select best candidate for unloading.

    Strategy: Pick the first model in the list.
    llama-swap returns model names as strings.
    """
    if not models:
        return None

    # Models are strings (model names)
    # Just pick the first one (oldest loaded)
    return models[0] if isinstance(models[0], str) else models[0].get('model')


def main():
    """Main monitoring loop."""
    config = get_config()

    logger.info("Starting pressure-aware model unloader")
    logger.info(f"API URL: {config['api_url']}")
    logger.info(f"Memory threshold: {config['memory_threshold']}%")
    logger.info(f"Check interval: {config['check_interval']}s")

    while True:
        try:
            memory_usage = get_memory_usage()

            if memory_usage > config['memory_threshold']:
                logger.warning(
                    f"Memory pressure detected: {memory_usage:.1f}% "
                    f"(threshold: {config['memory_threshold']}%)"
                )

                models = get_running_models(config['api_url'])

                if models:
                    candidate = select_unload_candidate(models)
                    if candidate:
                        logger.info(f"Attempting to unload: {candidate}")
                        unload_model(config['api_url'], candidate)
                    else:
                        logger.info("No suitable unload candidate found")
                else:
                    logger.info("No models currently loaded")
            else:
                logger.debug(f"Memory OK: {memory_usage:.1f}%")

        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")

        time.sleep(config['check_interval'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down pressure unloader")
        sys.exit(0)
