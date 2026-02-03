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
- [lm-studio-ollama-bridge](https://github.com/nicobrenner/lm-studio-ollama-bridge) (for syncing Ollama models)

## Installation

```bash
# Clone the repo
git clone https://github.com/youruser/model_serve.git
cd model_serve

# Install dependencies (will prompt for sudo to install llama-swap)
./install.sh

# Copy and edit environment config
cp .env.example .env
# Edit .env to set your BRIDGE_SCRIPT path

# Generate config from existing Ollama models
./model sync

# (Optional) Set up hourly auto-sync via cron
./setup_cron.sh
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
| `BRIDGE_SCRIPT` | (none) | Path to lm-studio-ollama-bridge |

### Custom Sampler Settings

Edit `custom_models.yaml` to configure per-model sampling:

```yaml
models:
  gemma3:27.4b-q8_0:
    sampler_args: "--top-nsigma 1.5 --top-k 0 --top-p 0.95 --temp 0.7"

  codestral:22.2b-q8_0:
    sampler_args: "--top-p 0.9 --temp 0.2"
    ttl: 3600  # Keep loaded longer
```

Custom settings are preserved when regenerating `config.yaml`.

## How It Works

1. **Ollama** downloads and manages model files
2. **lm-studio-ollama-bridge** creates symlinks in LM Studio's model directory
3. **generate_config.py** scans for models and creates `config.yaml`
4. **llama-swap** routes requests to the right model, spawning llama-server instances on demand
5. **pressure_unloader.py** monitors memory and unloads models when needed

## File Structure

```
model_serve/
├── model              # Main CLI (pull, rm, list, sync, run)
├── start.sh           # Start the server stack
├── stop.sh            # Stop all services
├── status.sh          # Check running status
├── install.sh         # Install dependencies
├── setup_cron.sh      # Set up hourly auto-sync
├── generate_config.py # Discover models → config.yaml
├── sync_bridge.sh     # Bridge + regenerate + reload
├── pressure_unloader.py # Memory pressure monitor
├── config.yaml        # Auto-generated llama-swap config
├── custom_models.yaml # Your custom sampler settings
├── .env.example       # Environment template
└── .env               # Your local config (git-ignored)
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

## License

MIT
