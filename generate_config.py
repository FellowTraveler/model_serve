#!/usr/bin/env python3
"""
Generate llama-swap config.yaml from available GGUF models.

Scans the LM Studio models directory (which contains Ollama symlinks)
and generates a config with all available models.

Custom sampler settings can be defined in custom_models.yaml which
will be merged in (overriding the auto-generated defaults).
"""

import os
import re
import yaml
from pathlib import Path


def get_config():
    """Load configuration from environment or defaults."""
    models_dir = os.environ.get('MODELS_DIR', '~/.cache/lm-studio/models')
    return {
        'models_dir': os.path.expanduser(models_dir),
        'llama_swap_port': os.environ.get('LLAMA_SWAP_PORT', '5847'),
        'model_ttl': int(os.environ.get('MODEL_TTL', '1800')),
        'default_ctx_size': int(os.environ.get('DEFAULT_CTX_SIZE', '8192')),
        'model_prefix': os.environ.get('MODEL_PREFIX', 'ls/'),  # Prefix to distinguish from Ollama
    }


def find_gguf_files(models_dir: str) -> list[tuple[str, str]]:
    """
    Find all GGUF files in the models directory.
    Returns list of (model_name, file_path) tuples.
    """
    models = []
    models_path = Path(models_dir)

    # Focus on ollama directory which has the symlinked models
    ollama_dir = models_path / 'ollama'

    if ollama_dir.exists():
        for model_dir in sorted(ollama_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            for gguf_file in sorted(model_dir.glob('*.gguf')):
                # Create clean model name
                model_id = create_model_name(model_dir.name, gguf_file.stem)
                models.append((model_id, str(gguf_file.resolve())))

    # Also scan other directories (non-ollama) for any GGUF files
    for publisher_dir in sorted(models_path.iterdir()):
        if not publisher_dir.is_dir():
            continue
        if publisher_dir.name == 'ollama':
            continue  # Already handled

        for model_dir in publisher_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for gguf_file in model_dir.glob('*.gguf'):
                # Use publisher/model format
                model_id = f"{publisher_dir.name}/{model_dir.name}"
                model_id = sanitize_model_id(model_id)

                # Avoid duplicates
                if not any(m[0] == model_id for m in models):
                    models.append((model_id, str(gguf_file.resolve())))

    return models


def create_model_name(dir_name: str, file_stem: str) -> str:
    """
    Create a clean model name from directory and file.
    Matches Ollama-style naming: model:variant
    """
    dir_lower = dir_name.lower()
    stem_lower = file_stem.lower()

    # If the file stem starts with the dir name, extract the variant
    # e.g., "gemma3/gemma3-12.2B-Q8_0" -> "gemma3:12.2b-q8_0"
    if stem_lower.startswith(dir_lower):
        variant = stem_lower[len(dir_lower):].lstrip('-_')
        if variant:
            return sanitize_model_id(f"{dir_lower}:{variant}")
        return sanitize_model_id(dir_lower)

    # Otherwise use dir:file format
    return sanitize_model_id(f"{dir_lower}:{stem_lower}")


def sanitize_model_id(name: str) -> str:
    """
    Convert a model name to a valid ID for llama-swap.
    Preserves colons for Ollama-style naming.
    """
    model_id = name.lower()

    # Replace invalid chars with hyphens (keep colons, dots, underscores)
    model_id = re.sub(r'[^a-z0-9\-_.:]', '-', model_id)

    # Collapse multiple hyphens
    model_id = re.sub(r'-+', '-', model_id)

    # Remove leading/trailing hyphens
    model_id = model_id.strip('-')

    return model_id


def estimate_ctx_size(model_path: str, default: int = 8192) -> int:
    """
    Estimate appropriate context size based on model size.
    Larger models may need smaller context to fit in memory.
    """
    try:
        size_gb = os.path.getsize(model_path) / (1024 ** 3)
    except:
        return default

    if size_gb > 50:
        return 2048
    elif size_gb > 30:
        return 4096
    else:
        return default


def load_custom_models(script_dir: Path) -> dict:
    """
    Load custom model settings from custom_models.yaml.
    Returns a dict of model_id -> settings to merge/override.
    """
    custom_file = script_dir / 'custom_models.yaml'

    if not custom_file.exists():
        return {}

    try:
        with open(custom_file) as f:
            data = yaml.safe_load(f) or {}
        return data.get('models') or {}
    except Exception as e:
        print(f"Warning: Failed to load custom_models.yaml: {e}")
        return {}


def generate_config(models: list[tuple[str, str]], config: dict, custom_models: dict) -> dict:
    """Generate llama-swap configuration dictionary."""

    llama_swap_config = {
        'listen': f"0.0.0.0:{config['llama_swap_port']}",
        'models': {}
    }

    prefix = config.get('model_prefix', '')

    for model_id, model_path in models:
        # Check for custom ctx_size first, otherwise estimate
        custom = custom_models.get(model_id)
        if custom and 'ctx_size' in custom:
            ctx_size = custom['ctx_size']
        else:
            ctx_size = estimate_ctx_size(model_path, config['default_ctx_size'])

        # Add prefix to distinguish from Ollama models
        prefixed_id = f"{prefix}{model_id}" if prefix else model_id

        # Default command
        cmd = f"llama-server --host 127.0.0.1 --port ${{PORT}} --model {model_path} --ctx-size {ctx_size}"

        model_config = {
            'cmd': cmd,
            'proxy': "http://127.0.0.1:${PORT}",
            'ttl': config['model_ttl'],
        }

        # Check for custom overrides (use original model_id for lookup)
        if model_id in custom_models:
            custom = custom_models[model_id]

            # Skip if entry has no value (just model name with nothing after)
            if custom is None:
                pass
            else:
                # Allow custom sampler args to be appended
                if 'sampler_args' in custom:
                    model_config['cmd'] = f"{cmd} {custom['sampler_args']}"

                # Allow full cmd override
                if 'cmd' in custom:
                    model_config['cmd'] = custom['cmd'].replace('${MODEL_PATH}', model_path)

                # Allow other overrides (ttl, etc)
                for key in ['ttl', 'proxy']:
                    if key in custom:
                        model_config[key] = custom[key]

        llama_swap_config['models'][prefixed_id] = model_config

    # Add any custom-only models (not auto-discovered)
    for model_id, custom in custom_models.items():
        if custom is None:
            continue
        if model_id not in llama_swap_config['models']:
            if 'cmd' in custom:
                llama_swap_config['models'][model_id] = {
                    'cmd': custom['cmd'],
                    'proxy': custom.get('proxy', "http://127.0.0.1:${PORT}"),
                    'ttl': custom.get('ttl', config['model_ttl']),
                }

    return llama_swap_config


def main():
    config = get_config()
    script_dir = Path(__file__).parent

    print(f"Scanning models in: {config['models_dir']}")

    models = find_gguf_files(config['models_dir'])
    print(f"Found {len(models)} models")

    custom_models = load_custom_models(script_dir)
    if custom_models:
        print(f"Loaded {len(custom_models)} custom model settings")

    if not models and not custom_models:
        print("No models found!")
        return

    llama_swap_config = generate_config(models, config, custom_models)

    # Write config
    output_path = script_dir / 'config.yaml'

    with open(output_path, 'w') as f:
        f.write("# Auto-generated llama-swap configuration\n")
        f.write("# Regenerate with: ./generate_config.py\n")
        f.write("# Custom settings go in: custom_models.yaml\n")
        f.write(f"# Models: {len(llama_swap_config['models'])}\n\n")
        yaml.dump(llama_swap_config, f, default_flow_style=False, sort_keys=False)

    print(f"Wrote config to: {output_path}")
    print(f"Total models: {len(llama_swap_config['models'])}")

    # Show sample models
    print("\nSample models:")
    for model_id, path in models[:15]:
        custom_marker = " [custom]" if model_id in custom_models else ""
        print(f"  {model_id}{custom_marker}")
    if len(models) > 15:
        print(f"  ... and {len(models) - 15} more")


if __name__ == '__main__':
    main()
