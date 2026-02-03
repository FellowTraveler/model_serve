# Model Serve

Multi-model LLM inference server for macOS Apple Silicon. Serves multiple models on a single OpenAI-compatible API endpoint with on-demand loading and automatic unloading.

## Why This Exists

**On-Demand + Keep-Loaded Behavior** - Loading large models (e.g., 120B) is expensive. We want models to load on first request and stay loaded until idle timeout or memory pressure—not reload on every request.

**Single API Port** - Simplifies client code (agents, web apps). No need to manage multiple ports for different models. Just specify the model name in the API request.

**Shared Model Directory** - Uses symlinked LM Studio models directory. No redundant storage—models downloaded via Ollama are served directly.

**Pressure-Aware Unloading** - TTL-based unloading isn't enough. When memory is high, automatically unload idle models to prevent OOM.

**Advanced Sampling** - Per-model control of top-n-sigma, min-p, temperature, etc. Different tasks benefit from different sampling behavior.

**Native macOS Execution** - No Docker overhead. Native builds on Apple Silicon (M1/M2/M3) for maximum throughput and memory efficiency.

## Features

- **Single API endpoint** - All models accessible via one port (OpenAI-compatible)
- **On-demand loading** - Models load when first requested, not at startup
- **Auto-unload** - Models unload after idle timeout (TTL) or memory pressure
- **Ollama integration** - Pull/remove models with automatic config sync
- **Custom samplers** - Configure per-model sampling parameters (ctx_size, temperature, etc.)

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

# (Optional) Set up cron keep-alive and start server
./setup_cron.sh start
```

If you cloned without `--recursive`:
```bash
git submodule update --init --recursive
./install.sh
```

## Usage

### Server Control

```bash
# Start the server (runs in background, logs to model_serve.log)
./model start

# Check if services are running
./model status

# Stop all services
./model stop
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

# Show running models with resource usage
./model stats

# Run a model interactively
./model run gemma3:12b
```

The `stats` command shows all loaded models with context size, slots, state, and system memory:
```
======================================================================
MODEL                                              CTX  SLOTS    STATE
======================================================================
ls/gemma3:27.4b-q8_0                            32,768      1    ready
ls/codestral:22.2b-q8_0                         32,768      1    ready
======================================================================
TOTAL                                           65,536      2

System Memory: 65.2% used (83.5GB / 128.0GB)
```

### API Endpoints

```bash
# Chat completion (OpenAI-compatible)
curl http://127.0.0.1:5847/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ls/gemma3:12.2b-q8_0", "messages": [{"role": "user", "content": "Hello"}]}'

# List available models
curl http://127.0.0.1:5847/v1/models

# List currently loaded models (with full details)
curl http://127.0.0.1:5847/running

# Unload a specific model
curl -X POST "http://127.0.0.1:5847/unload?model=ls/gemma3:12.2b-q8_0"
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
| `MEMORY_PRESSURE_THRESHOLD` | 75 | Memory % to trigger auto-unload (increase to 85+ for 128GB systems) |
| `PRESSURE_CHECK_INTERVAL` | 30 | Seconds between memory pressure checks |
| `MODELS_DIR` | ~/.cache/lm-studio/models | Where models are stored |
| `BRIDGE_SCRIPT` | (bundled) | Path to lm-studio-ollama-bridge (uses submodule by default) |
| `MODEL_PREFIX` | `ls/` | Prefix for model names to distinguish from Ollama in Open WebUI |
| `DEFAULT_CTX_SIZE` | 8192 | Default context size for models without custom settings |
| `DEFAULT_PARALLEL` | 1 | Slots per model (1 = single-user, saves memory; 4 = multi-user) |
| `BRIDGE_SYNC_INTERVAL` | 3600 | Seconds between automatic syncs (when using start.sh) |

### Model Naming

Models are prefixed with `ls/` by default so you can distinguish them from Ollama models in Open WebUI:
- **llama-swap**: `ls/gemma3:12.2b-q8_0`
- **Ollama**: `hf.co/mradermacher/Gemma-3-27B-Derestricted-GGUF:q8_0`

Change the prefix in `.env` by setting `MODEL_PREFIX` (or set to empty to disable).

### Custom Model Settings (Optional)

Custom settings are **optional** - the defaults work fine for most models. Add them later if you want to fine-tune specific models.

Edit `custom_models.yaml` to configure per-model settings:

```yaml
models:
  # Use model name WITHOUT the ls/ prefix
  # General model with adaptive sampling
  gemma3:27.4b-q8_0:
    sampler_args: "--top-nsigma 1.5 --min-p 0.05 --temp 1.2"
    ctx_size: 32768  # Override default context size

  # Coding model with lower temperature
  codestral:22.2b-q8_0:
    sampler_args: "--top-nsigma 1.5 --min-p 0.05 --temp 0.4"
    ctx_size: 32768
    ttl: 3600  # Keep loaded longer for coding sessions

  # Large context model
  gpt-oss-120b-derestricted-gguf:117b-unknown:
    sampler_args: "--top-nsigma 1.5 --min-p 0.05 --temp 1.2"
    ctx_size: 65536  # 64k context (model supports 128k)
```

Available settings:
- `sampler_args`: Additional args appended to llama-server command
- `ctx_size`: Context size (overrides auto-estimated value based on model size)
- `ttl`: Custom idle timeout in seconds
- `cmd`: Full command override (use `${MODEL_PATH}` and `${PORT}` placeholders)

**Sampler notes:**
- `--top-nsigma 1.5`: Adaptive sampling based on logit std deviation (works well with higher temps)
- `--min-p 0.05`: Filters tokens below 5% of top token probability
- `--temp`: Temperature (1.0-1.2 for general, 0.7 for Mistral, 0.4 for coding)

After editing, run `./model sync` to regenerate config. If llama-swap is running and the config changed, it will automatically restart to pick up the new settings.

### Run as System Service (Recommended)

For persistent operation that survives logout and auto-starts on boot:

```bash
# Install and start the service
./setup_service.sh install

# Check status
./setup_service.sh status

# Stop/start the service
./setup_service.sh stop
./setup_service.sh start

# Uninstall completely
./setup_service.sh uninstall
```

This uses **launchd** on macOS and **systemd** on Linux. The service auto-restarts if it crashes and starts on login.

### Keep-Alive with Cron (Fallback)

If system services don't work for your setup, use cron as a fallback:

```bash
# Install cron job AND start server now
./setup_cron.sh start

# Check current status
./setup_cron.sh

# Remove cron job AND stop server
./setup_cron.sh stop
```

The cron job runs `./model start` hourly. If the server is already running, it exits immediately. If not, it starts the full stack.

**Note:** Cron jobs may not survive logout on some systems. Use `setup_service.sh` for reliable persistence.

## How It Works

1. **Ollama** downloads and manages model files
2. **lm-studio-ollama-bridge** (bundled submodule) creates symlinks in LM Studio's model directory
3. **generate_config.py** scans for models, merges custom settings, and creates `config.yaml`
4. **llama-swap** routes requests to the right model, spawning llama-server instances on demand
5. **pressure_unloader.py** monitors memory and unloads idle models when pressure is high
6. **sync_loop.sh** periodically syncs models and restarts llama-swap if config changed

## File Structure

```
model_serve/
├── model                      # Main CLI (pull, rm, list, sync, stats, start, stop, status)
├── start.sh                   # Start the server stack
├── stop.sh                    # Stop all services
├── status.sh                  # Check running status
├── install.sh                 # Install dependencies
├── setup_service.sh           # Install as system service (launchd/systemd)
├── setup_cron.sh              # Cron-based keep-alive (fallback)
├── generate_config.py         # Discover models → config.yaml
├── sync_bridge.sh             # Sync wrapper with PATH setup (used by sync_loop.sh)
├── sync_loop.sh               # Periodic sync loop (used by start.sh)
├── pressure_unloader.py       # Memory pressure monitor
├── lm-studio-ollama-bridge/   # Submodule: Ollama-LM Studio sync tool
├── config.yaml                # Auto-generated llama-swap config (git-ignored)
├── custom_models.yaml         # Your custom model settings
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

**Models keep getting unloaded (memory pressure):**

If you have plenty of RAM but models keep unloading, raise the memory threshold in `.env`:
```bash
MEMORY_PRESSURE_THRESHOLD=85  # or higher for 128GB+ systems
```
Then restart the server.

**Check what's running:**
```bash
./model stats    # Show loaded models with resource usage
./status.sh      # Check all services
```

## License

MIT

## See my AGI Articles:

### [Pondering AGI](https://christopherdavidodom.substack.com/p/pondering-agi)
[![Pondering AGI](https://substackcdn.com/image/fetch/w_600,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fed39229d-fefd-4030-8b62-52f8cb2b0f05_1024x768.jpeg)](https://christopherdavidodom.substack.com/p/pondering-agi)

### [Pondering AGI Part 2](https://christopherdavidodom.substack.com/p/pondering-agi-part-2)
[![Pondering AGI Part 2](https://substackcdn.com/image/fetch/w_600,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6815d224-5ae0-4e71-bd50-f14c3525cce9_725x522.png)](https://christopherdavidodom.substack.com/p/pondering-agi-part-2)

### [Pondering AGI Part 3](https://christopherdavidodom.substack.com/p/pondering-agi-part-3)
[![Pondering AGI Part 3](https://substackcdn.com/image/fetch/$s_!ooN_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F504d2f57-a02f-4313-b76e-aa279783df7f_796x568.png)](https://christopherdavidodom.substack.com/p/pondering-agi-part-3)

