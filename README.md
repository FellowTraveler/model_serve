# Model Serve

Multi-model LLM inference server for macOS Apple Silicon. Serves multiple models on a single OpenAI-compatible API endpoint with on-demand loading and automatic unloading.

## Features

- **Single API endpoint** - All models accessible via one port (OpenAI-compatible)
- **On-demand loading** - Models load when first requested, not at startup
- **Auto-unload** - Models unload after idle timeout (TTL) or memory pressure
- **Ollama integration** - Pull/remove models with automatic config sync
- **Custom samplers** - Configure per-model sampling parameters

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- [Homebrew](https://brew.sh)
- [Ollama](https://ollama.ai) installed

## Installation

```bash
# Clone the repo with submodules
git clone --recursive https://github.com/FellowTraveler/model_serve.git
cd model_serve

# Install dependencies (will prompt for sudo to install llama-swap)
./install.sh

# Generate config from existing Ollama models
./model sync

# (Optional) Set up hourly auto-sync via cron
./setup_cron.sh
```

If you cloned without `--recursive`:
```bash
git submodule update --init --recursive
./install.sh
```

## Usage

### Start the Server

```bash
./start.sh
```

The server runs on port 5847 by default (configurable in `.env`).

### Model Management

```bash
# Pull a new model (auto-syncs config)
./model pull gemma3:27b

# Remove a model
./model rm gemma3:27b

# List all models
./model list

# List models matching a filter (case-insensitive)
./model list qwen
./model list derestricted
./model list 70b

# Manual sync (after direct ollama commands)
./model sync

# Run a model interactively
./model run gemma3:12b
```

### API Endpoints

```bash
# Chat completion (OpenAI-compatible)
curl http://127.0.0.1:5847/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:12.2b-q8_0", "messages": [{"role": "user", "content": "Hello"}]}'

# List available models
curl http://127.0.0.1:5847/v1/models

# List currently loaded models
curl http://127.0.0.1:5847/running

# Unload a specific model
curl -X POST http://127.0.0.1:5847/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:12.2b-q8_0"}'
```

### Stop the Server

```bash
./stop.sh
```

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_SWAP_PORT` | 5847 | API server port |
| `MODEL_TTL` | 1800 | Idle timeout in seconds (30 min) |
| `MEMORY_PRESSURE_THRESHOLD` | 75 | Memory % to trigger auto-unload |
| `MODELS_DIR` | ~/.cache/lm-studio/models | Where models are stored |
| `BRIDGE_SCRIPT` | (bundled) | Path to lm-studio-ollama-bridge (uses submodule by default) |
| `MODEL_PREFIX` | `ls/` | Prefix for model names to distinguish from Ollama in Open WebUI |

### Model Naming

Models are prefixed with `ls/` by default so you can distinguish them from Ollama models in Open WebUI:
- **llama-swap**: `ls/gemma3:12.2b-q8_0`
- **Ollama**: `hf.co/mradermacher/Gemma-3-27B-Derestricted-GGUF:q8_0`

Change the prefix in `.env` by setting `MODEL_PREFIX` (or set to empty to disable).

### Custom Sampler Settings (Optional)

Custom sampler settings are **optional** - the defaults work fine for most models. Add them later if you want to fine-tune specific models (e.g., lower temperature for coding).

Edit `custom_models.yaml` to configure per-model sampling:

```yaml
models:
  # Use model name WITHOUT the ls/ prefix
  gemma3:27.4b-q8_0:
    sampler_args: "--top-nsigma 1.5 --top-k 0 --top-p 0.95 --temp 0.7"

  codestral:22.2b-q8_0:
    sampler_args: "--top-p 0.9 --temp 0.2"
    ttl: 3600  # Keep loaded longer
```

After editing, run `./model sync` to regenerate config. The server auto-reloads.

Custom settings are preserved when regenerating `config.yaml`.

### Auto-Sync with Cron (Optional)

If you use `./model pull` and `./model rm`, you don't need cron - those commands auto-sync.

However, if you sometimes use `ollama pull` directly, set up hourly auto-sync:

```bash
# Install cron job
./setup_cron.sh

# Verify it's installed
crontab -l | grep sync_bridge

# Remove if no longer needed
crontab -l | grep -v sync_bridge | crontab -
```

The cron job runs `sync_bridge.sh` hourly, which syncs Ollama models and regenerates the config.

## How It Works

1. **Ollama** downloads and manages model files
2. **lm-studio-ollama-bridge** (bundled submodule) creates symlinks in LM Studio's model directory
3. **generate_config.py** scans for models and creates `config.yaml`
4. **llama-swap** routes requests to the right model, spawning llama-server instances on demand
5. **pressure_unloader.py** monitors memory and unloads models when needed

## File Structure

```
model_serve/
├── model                      # Main CLI (pull, rm, list, sync, run)
├── start.sh                   # Start the server stack
├── stop.sh                    # Stop all services
├── status.sh                  # Check running status
├── install.sh                 # Install dependencies
├── setup_cron.sh              # Set up hourly auto-sync
├── generate_config.py         # Discover models → config.yaml
├── sync_bridge.sh             # Bridge + regenerate + reload
├── pressure_unloader.py       # Memory pressure monitor
├── lm-studio-ollama-bridge/   # Submodule: Ollama-LM Studio sync tool
├── config.yaml                # Auto-generated llama-swap config
├── custom_models.yaml         # Your custom sampler settings
├── .env.example               # Environment template
└── .env                       # Your local config (git-ignored)
```

## Troubleshooting

**Models not appearing after `ollama pull`:**
```bash
./model sync
```

**Server won't start:**
```bash
# Check if port is in use
lsof -i :5847

# Check llama-swap is installed
which llama-swap
```

**Model loading fails:**
```bash
# Check the model file exists
ls -la ~/.cache/lm-studio/models/ollama/<model-name>/
```

**Bridge not built:**
```bash
./install.sh
```

## License

MIT
