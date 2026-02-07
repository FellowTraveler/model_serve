# Model Serve

A wrapper around [llama-swap](https://github.com/mostlygeek/llama-swap) that manages Ollama models on a single OpenAI-compatible API endpoint. Designed for users who want to serve multiple models simultaneously without duplicate storage, with access to advanced sampler settings like min-p and top-n-sigma (top-σ). Automatically symlinks Ollama models into LM Studio's models folder (or any directory you configure).

Works on **macOS** (Intel and Apple Silicon) and **Linux**.

## Why This Exists

**No Duplicate Models** - If you use both Ollama and LM Studio, you don't want two copies of every model. This project syncs Ollama models to a shared directory via symlinks.

**Single API Port** - Serve all your models on one port. Simplifies client code (agents, web apps). Just specify the model name in the API request.

**On-Demand Loading** - Models load on first request and stay loaded until idle timeout or memory pressure. Large models (120B+) don't reload on every request.

**Pressure-Aware Unloading** - TTL-based unloading isn't enough. When memory is high, automatically unload idle models to prevent OOM.

**Advanced Sampling** - Per-model control of min-p, top-n-sigma (top-σ), temperature, etc. Different tasks benefit from different sampling behavior.

## Features

- **Single API endpoint** - All models accessible via one port (OpenAI-compatible)
- **GPT-OSS Harmony support** - Automatic Harmony encoding/decoding for GPT-OSS models, including tool calls
- **MXFP4 model support** - Models with MXFP4 quantization automatically route to Ollama (llama.cpp doesn't support MXFP4)
- **Embeddings support** - Embedding models (containing "embed" in name) route to Ollama for vector generation
- **On-demand loading** - Models load when first requested, not at startup
- **LRU-based auto-unload** - Models unload after idle timeout (TTL) or memory pressure, with least-recently-used priority
- **Ollama integration** - Pull/remove models with automatic config sync
- **LM Studio integration** - (Optional) Ollama models are symlinked into LM Studio
- **Advanced samplers** - Per-model min-p, top-n-sigma (top-σ), temperature, ctx_size

## Prerequisites

- **macOS** (Intel or Apple Silicon) or **Linux**
- [Ollama](https://ollama.ai) - required for pulling and managing models
- **Python 3** with pip - for the management scripts
- [Homebrew](https://brew.sh) (macOS) - used by `install.sh` to install llama.cpp and Go
- **Go compiler** - required to build lm-studio-ollama-bridge (installed via Homebrew if missing)
- **Optional:** [LM Studio](https://lmstudio.ai) - if installed, Ollama models appear in both UIs

**Note:** On Linux, install Go and [llama.cpp](https://github.com/ggerganov/llama.cpp) before running `install.sh`. The script will download the correct llama-swap binary automatically and build the bridge from source.

## Installation

```bash
# Clone the repo with submodules
git clone --recursive https://github.com/FellowTraveler/model_serve.git
cd model_serve

# Install dependencies (will prompt for sudo to install llama-swap)
./install.sh

# Generate config from existing Ollama models
./model sync

# (Optional) Install as system service for auto-start on boot
./setup_service.sh install   # Recommended: launchd (macOS) or systemd (Linux)
# OR
./setup_cron.sh start        # Fallback: hourly cron job
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

The Harmony proxy listens on port 5846 by default - this is the client-facing endpoint. All clients should connect here regardless of model. The proxy automatically handles Harmony encoding/decoding for GPT-OSS models and passes through all other models unchanged.

**MXFP4 Models:** Models with "MXFP4" in the name (case-insensitive) are automatically routed to Ollama instead of llama-swap, since llama.cpp doesn't support MXFP4 quantization. This happens transparently - just use the model name as usual.

**Embedding Models:** Models with "embed" in the name are routed to Ollama for the `/v1/embeddings` endpoint. Available models include `nomic-embed-text` (768 dimensions) and `mxbai-embed-large` (1024 dimensions). Pull them with `ollama pull nomic-embed-text` and `ollama pull mxbai-embed-large`.

### Model Management

```bash
# Pull a new model (auto-syncs config)
./model pull gemma3:27b

# Remove a model
./model rm gemma3:27b

# List Ollama models (with optional filter)
./model list
./model list qwen

# List llama-swap model names from config
./model list --config
./model list --config gemma

# Show Ollama → llama-swap name mapping
./model names gemma

# Show full config for a model (all settings)
./model show gemma3:12

# Manual sync (after direct ollama commands)
./model sync

# Show running models with resource usage
./model stats

# Run a model interactively
./model run gemma3:12b
```

**Finding model names:** Ollama model names (e.g., `gemma3:1b-it-qat`) don't always match llama-swap names (e.g., `gemma3:999.89m-q4_0`) because llama-swap uses the actual GGUF file metadata. Use `./model names` or `./model list --config` to find the correct names for API requests and `custom_models.yaml`.

The `stats` command shows all loaded models with context size, slots, state, and system memory:
```
======================================================================
MODEL                                              CTX  SLOTS    STATE
======================================================================
ms/gemma3:27.4b-q8_0                            32,768      1    ready
ms/codestral:22.2b-q8_0                         32,768      1    ready
======================================================================
TOTAL                                           65,536      2

System Memory: 65.2% used (83.5GB / 128.0GB)
```

### API Endpoints

Connect to port 5846 (Harmony proxy) for all models. The proxy handles Harmony encoding/decoding for GPT-OSS models automatically and passes through all other models unchanged.

```bash
# Chat completion (OpenAI-compatible) - works with any model
curl http://127.0.0.1:5846/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ms/gemma3:12.2b-q8_0", "messages": [{"role": "user", "content": "Hello"}]}'

# GPT-OSS models work the same way - Harmony handled automatically
# Tool calls are fully supported for agentic workflows
curl http://127.0.0.1:5846/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ms/gpt-oss-20b-derestricted-gguf-20.9b-q8_0", "messages": [{"role": "user", "content": "Hello"}]}'

# MXFP4 models route to Ollama automatically (llama.cpp doesn't support MXFP4)
curl http://127.0.0.1:5846/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hf.co/Felladrin/gguf-MXFP4-gpt-oss-20b-Derestricted:latest", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings - models with "embed" in name route to Ollama
curl http://127.0.0.1:5846/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "ms/nomic-embed-text-137m-f16", "input": "Hello world"}'

# List available models
curl http://127.0.0.1:5846/v1/models

# Health check
curl http://127.0.0.1:5846/health

# Direct llama-swap access (port 5847) - for debugging only
curl http://127.0.0.1:5847/running
curl -X POST "http://127.0.0.1:5847/unload?model=ms/gemma3:12.2b-q8_0"
```

## Configuration

### Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `HARMONY_PROXY_PORT` | 5846 | Client-facing API port (connects here) |
| `LLAMA_SWAP_PORT` | 5847 | Internal llama-swap port (proxy forwards here) |
| `OLLAMA_BASE` | http://localhost:11434 | Ollama backend URL for MXFP4 models |
| `MODEL_TTL` | 1800 | Idle timeout in seconds (30 min) |
| `MEMORY_PRESSURE_THRESHOLD` | 75 | Memory % to trigger auto-unload (increase to 85+ for 128GB systems) |
| `PRESSURE_CHECK_INTERVAL` | 30 | Seconds between memory pressure checks |
| `MIN_MODEL_AGE_SECONDS` | 5 | Minimum idle time before a model can be unloaded (protects recently-used models) |
| `MODELS_DIR` | ~/.cache/lm-studio/models | Where model symlinks are created (see below) |
| `BRIDGE_SCRIPT` | (bundled) | Path to lm-studio-ollama-bridge (uses submodule by default) |
| `MODEL_PREFIX` | `ms/` | Prefix for model names to distinguish from Ollama in Open WebUI |
| `DEFAULT_CTX_SIZE` | 8192 | Default context size for models without custom settings |
| `DEFAULT_PARALLEL` | 1 | Slots per model (1 = single-user, saves memory; 4 = multi-user) |
| `BRIDGE_SYNC_INTERVAL` | 3600 | Seconds between automatic syncs (when using start.sh) |
| `HARMONY_MODELS_CONFIG` | harmony_models.yaml | Path to Harmony models config file |

### Models Directory Structure

The bridge syncs Ollama models to `MODELS_DIR` with this structure:

```
MODELS_DIR/
└── ollama/
    ├── gemma3/
    │   └── gemma3-12.2B-Q8_0.gguf  → (symlink to Ollama blob)
    ├── codestral/
    │   └── codestral-22.2B-Q8_0.gguf  → (symlink to Ollama blob)
    └── ...
```

**LM Studio is optional.** The default `MODELS_DIR` is LM Studio's cache directory, so if you have LM Studio installed, synced models appear in both UIs automatically. But you can set `MODELS_DIR` to any directory—the bridge creates the required structure.

### Model Naming

Models are prefixed with `ms/` (model_serve) by default so you can distinguish them from Ollama models in Open WebUI:
- **model_serve**: `ms/gemma3:12.2b-q8_0`
- **Ollama**: `hf.co/mradermacher/Gemma-3-27B-Derestricted-GGUF:q8_0`

Change the prefix in `.env` by setting `MODEL_PREFIX` (or set to empty to disable).

### Custom Model Settings (Optional)

Custom settings are **optional** - the defaults work fine for most models. Add them later if you want to fine-tune specific models.

Edit `custom_models.yaml` to configure per-model settings:

```yaml
models:
  # Use model name WITHOUT the ms/ prefix (use ./model names to find names)
  # Standard settings
  gemma3:12.2b-q8_0:
    sampler_args: "--top-nsigma 1.5 --min-p 0.05 --temp 1.0"
    ctx_size: 32768

  # Alias: same model file, different settings (appears as ms/gemma3-creative)
  # Useful for having "creative" vs "precise" variants of the same model
  gemma3-creative:
    base_model: gemma3:12.2b-q8_0
    sampler_args: "--top-nsigma 2.0 --min-p 0.02 --temp 1.4"
    ctx_size: 8192  # Smaller context = less memory

  # Coding model with lower temperature
  codestral:22.2b-q8_0:
    sampler_args: "--top-nsigma 1.5 --min-p 0.05 --temp 0.4"
    ctx_size: 32768
    ttl: 3600  # Keep loaded longer for coding sessions
```

Available settings:
- `sampler_args`: Additional args appended to llama-server command
- `ctx_size`: Context size (overrides auto-estimated value based on model size)
- `ttl`: Custom idle timeout in seconds
- `base_model`: Create an alias pointing to another model's file (same GGUF, different settings)
- `cmd`: Full command override (use `${MODEL_PATH}` and `${PORT}` placeholders)

**Sampler args** (passed to llama-server via `sampler_args`):

| Arg | Default | Description |
|-----|---------|-------------|
| `--temp` | 0.8 | Temperature (1.0 balanced, 1.4 creative, 0.4 coding) |
| `--top-k` | 40 | Top-k sampling (0 = disabled) |
| `--top-p` | 0.95 | Top-p / nucleus sampling (1.0 = disabled) |
| `--min-p` | 0.05 | Min-p sampling - filters tokens below X% of top token prob (0.0 = disabled) |
| `--top-nsigma` | -1 | Top-n-sigma - adaptive sampling based on logit std deviation (-1 = disabled, try 1.5-2.0) |
| `--repeat-penalty` | 1.0 | Penalize repeated tokens (1.0 = disabled) |
| `--presence-penalty` | 0.0 | Presence penalty for repetition (0.0 = disabled) |
| `--frequency-penalty` | 0.0 | Frequency penalty for repetition (0.0 = disabled) |
| `--reasoning-budget` | -1 | For thinking models: -1 = unlimited, 0 = disable thinking |
| `--reasoning-format` | none | For thinking models: `deepseek` extracts thoughts to `reasoning_content` |

**Recommended combinations:**
- **Balanced**: `--top-nsigma 1.5 --min-p 0.05 --temp 1.0` (top-p/top-k disabled)
- **Creative**: `--top-nsigma 2.0 --min-p 0.02 --temp 1.4`
- **Coding**: `--top-nsigma 1.5 --min-p 0.05 --temp 0.4`
- **Mistral models**: Use `--temp 0.7` (these models prefer lower temperature)

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

1. **Ollama** downloads and manages model files (stored as blobs)
2. **lm-studio-ollama-bridge** (bundled) creates symlinks in `MODELS_DIR` pointing to Ollama blobs
3. **generate_config.py** scans `MODELS_DIR` for GGUF files, merges custom settings, creates `config.yaml`
4. **harmony_proxy.py** receives all API requests on port 5846:
   - GPT-OSS models: Harmony encoding/decoding with tool call support
   - MXFP4 models: Routes to Ollama (llama.cpp doesn't support MXFP4)
   - Other models: Passes through to llama-swap unchanged
5. **llama-swap** routes requests to the right model, spawning llama-server instances on demand
6. **pressure_unloader.py** monitors memory and unloads idle models using LRU (least recently used) strategy across both llama-swap and Ollama backends
7. **sync_loop.sh** periodically re-syncs models and restarts llama-swap if config changed

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
├── harmony_proxy.py           # Client-facing proxy (Harmony, MXFP4 routing, tool calls)
├── generate_config.py         # Discover models → config.yaml
├── sync_bridge.sh             # Sync wrapper with PATH setup (used by sync_loop.sh)
├── sync_loop.sh               # Periodic sync loop (used by start.sh)
├── pressure_unloader.py       # LRU-based memory pressure monitor (llama-swap + Ollama)
├── harmony_models.yaml        # List of models requiring Harmony encoding
├── lm-studio-ollama-bridge/   # Submodule: Ollama-LM Studio sync tool
├── config.yaml                # Auto-generated llama-swap config (git-ignored)
├── custom_models.yaml         # Your custom model settings
├── .env.example               # Environment template
└── .env                       # Your local config (git-ignored)
```

## Troubleshooting

**Models not appearing after `ollama pull`:**

Use `./model pull` instead of `ollama pull` to automatically sync:
```bash
# Pull from Ollama registry
./model pull glm-4.7-flash:q4_K_M

# Pull from HuggingFace (use hf.co/ prefix)
./model pull hf.co/unsloth/GLM-4.7-Flash-GGUF:Q8_K_XL
```

If you already used `ollama pull` directly, run sync manually:
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
# Check the model symlink exists (default MODELS_DIR shown)
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

