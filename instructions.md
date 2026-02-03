Here’s a concise, clear statement of everything you’ve discussed and are trying to achieve so your local coding agent understands your goals and constraints. You can paste this at the very top of your existing markdown setup doc.

⸻

📋 Project Requirements and Rationale

🎯 Overall Goal

We want to set up a local multi-model LLM inference server that:
    1.    Serves multiple different models concurrently on a single API endpoint/port (OpenAI-style).
This includes large chat models (e.g., GPT-OSS 20B/120B in MXFP4), embedding models, and smaller models (e.g., Gemma3-27B, Qwen3-4B), all accessible via model in the API request.
    2.    Loads models on demand when the first request for a given model arrives, rather than pre-loading all models simultaneously.
This minimizes startup time and memory pressure.
    3.    Keeps models loaded in memory after use so subsequent calls do not pay the reload cost — unless the model has been idle beyond a configured time period (TTL) or the system is under memory pressure.
    4.    Unloads models automatically not only after idle timeout (TTL) but also when system resources are strained (pressure-aware unloading).
    5.    Is able to reuse the existing LM Studio models directory (~/.cache/lm-studio/models) as the canonical model storage location, so models downloaded via Ollama and symlinked into LM Studio are served directly without duplication.
    6.    Supports advanced sampling configurations, including settings like top-n-sigma, top-k, top-p, temperature, and related flags passed to the underlying inference server (llama-server) so we can fine-tune sampling behavior.
    7.    Works natively on macOS Apple Silicon (M3 Max) without container overhead — because native performance is crucial.
    8.    Integrates the symlink bridge workflow (the script at /Users/au/src/lm-studio-ollama-bridge/lm-studio-ollama-bridge) on a periodic basis so that models downloaded/removed by Ollama remain available in the LM Studio directory and, by extension, to the model server.
    9.    Provides introspection and management endpoints (e.g., lists currently loaded models, allows manual unloads) so monitoring and auxiliary scripts can make unload decisions.
    10.    Minimizes unexpected or unnecessary unloading, meaning:

    •    Idle TTL unloads should only happen after a configured idle period.
    •    Pressure-aware unloads should be triggered transparently when system memory or other key resources are stressed (e.g., high unified memory usage).

    11.    Should not force unloading all models every time a different model is requested.
While some router/proxy systems swap models by unloading currently running ones when a new model is requested, our setup must maintain multiple loaded models concurrently if they are in use or within TTL. This avoids performance degradation from repeated reloads and aligns with common multi-model usage patterns.  ￼
    12.    Allows custom scripts to interact with the server (e.g., to call /running or /models/unload endpoints) to implement pressure-aware logic or maintenance tasks.

⸻

📌 Why These Requirements Matter

🔹 On-Demand + Keep-Loaded Behavior

Loading big models (e.g., 120B class) is expensive — doing it on every request destroys performance.
We want the peak memory cost only when necessary and retention until idle threshold or pressure eviction.  ￼

🔹 Single API Port

This simplifies client code (e.g., agent, web apps) — no need to manage multiple ports and backends for different models. Systems like llama-swap provide OpenAI API compatibility while routing to correct backends based on the "model" field.  ￼

🔹 Shared Model Directory

Using the symlinked LM Studio models directory means:
    •    No redundant storage or double downloads,
    •    Single point of truth for available models,
    •    Seamless integration with your existing tools (Ollama, LM Studio). This directory already contains all models you care about, and symlinks should stay updated via the bridge script.

🔹 Pressure-Aware Unloading

Idle TTL is time-based — helpful, but not enough. If the system is under high memory usage, we want a programmatic way to unload least-important models first to avoid out-of-memory conditions, while respecting idle TTL as well.

🔹 Advanced Sampling

Different tasks (chat, summarization, QA) benefit from different sampling behaviors. Support for parameters like:
    •    top_n_sigma,
    •    top_k,
    •    top_p,
    •    min_p,
    •    temperature,
ensures balanced quality and creativity. These need to be passed through to the actual inference engine (llama-server) from the routing layer.

🔹 Native macOS Execution

Containerization (Docker) can work but is not appropriate for Apple Silicon native performance on M3 Max — we explicitly require native builds of routers and inference servers to maximize throughput and memory efficiency.

🔹 Monitoring and Manual Control

Exposing endpoints like /running and /models/unload is necessary so auxiliary scripts (e.g., memory watchers) can make informed decisions about resource usage and corrective actions.

⸻

🧠 Summary

In short, the system must:
    •    Serve multiple models on one port,
    •    Load models on first request and keep them loaded,
    •    Auto-unload idle or when memory is tight,
    •    Leverage existing LM Studio/Ollama model directories,
    •    Allow advanced sampling configuration,
    •    Be natively runnable on macOS (M3 Max),
    •    Provide scriptable control for resource management.

⸻

Let me know if you want this formatted as YAML comments or integrated into an automated task script for your coding agent.


Here’s a polished Markdown setup guide you can use as a single artifact (save as instructions.md) for a local coding agent or personal reference. It assumes your models are stored under your LM Studio models directory (~/.cache/lm-studio/models) and that you use your Ollama→LM Studio symlinker regularly.

⸻

🧠 Multi-Model LLM Serving with llama-server + llama-swap

On Demand Load + Idle/Pressure-Aware Unload • Custom Sampling • Shared LM Studio Model Directory

⸻

🛠 1. System & Prerequisites
    •    Platform: macOS (Apple Silicon M-series)
    •    Tools used: Homebrew, Python (for monitoring script), llama-server (from llama.cpp), llama-swap proxy, your symlink bridge (lm-studio-ollama-bridge)
    •    Model location: ~/.cache/lm-studio/models/... (your existing structure, e.g., ~/.cache/lm-studio/models/adrienbrault/..., etc.)
    •    Model format: GGUF files inside cherry-picked subfolders per LM Studio convention (two-level structure)  ￼

⸻

📦 2. Install llama-server (Inference Backend)

This runs the actual model in an OpenAI-compatible HTTP server.

✔ macOS (Homebrew)

brew install llama.cpp

This gives you llama-server, which is the HTTP front-end for the llama.cpp engine.  ￼

⸻

🚀 3. Install llama-swap (Multi-Model Router)

llama-swap is a lightweight Go binary that:

✅ Listens on one API port
✅ Loads the requested model on demand
✅ Keeps it loaded until idle timeout (ttl)
✅ Supports manual /models/unload via API
✅ Works with any OpenAI-compatible server (like llama-server)  ￼

✔ macOS Install

brew install llama-swap

Or download a release binary from the mostlygeek/llama-swap GitHub repo.  ￼

⸻

⚙️ 4. Create llama-swap Configuration

Save as config.yaml:

listen: 0.0.0.0:5005

models:
  gpt-oss-20b:
    cmd: >
      llama-server --host 127.0.0.1 --port ${PORT}
      --model /Users/au/.cache/lm-studio/models/hf.co/<PATH>/gpt-oss-20b-Derestricted-MXFP4.gguf
    proxy: "http://127.0.0.1:${PORT}"
    ttl: 1800   # idle unload after 30 min

  gemma3-27b:
    cmd: >
      llama-server --host 127.0.0.1 --port ${PORT}
      --model /Users/au/.cache/lm-studio/models/<PATH>/gemma3-27b.gguf
    proxy: "http://127.0.0.1:${PORT}"
    ttl: 1800

  qwen3-4b:
    cmd: >
      llama-server --host 127.0.0.1 --port ${PORT}
      --model /Users/au/.cache/lm-studio/models/<PATH>/qwen3-4b-*.gguf
    proxy: "http://127.0.0.1:${PORT}"
    ttl: 1800

Notes:
✔ ttl sets idle unload (seconds after the last inference).
✔ cmd should point to the GGUF files in your LM Studio cache.  ￼

Start the router:

llama-swap --config config.yaml


⸻

🧠 5. Pressure-Aware Unloading Script

Because llama-swap only unloads by TTL, you can add a monitor that also unloads if memory is high.

Create pressure_unloader.py

#!/usr/bin/env python3
import requests, psutil, time

API="http://127.0.0.1:5005"

def running_models():
    return requests.get(f"{API}/running").json()

def unload_mod(name):
    requests.post(f"{API}/models/unload", json={"model":name})

while True:
    mem = psutil.virtual_memory().percent
    if mem > 75:  # adjust
        models = running_models()
        if models:
            # choose a candidate (last, oldest, etc.)
            candidate = models[-1]["model"]
            unload_mod(candidate)
    time.sleep(30)

Run it in the background alongside llama-swap.

⸻

📁 6. Integrating Your Symlink Bridge

You already use:

/Users/au/src/lm-studio-ollama-bridge/lm-studio-ollama-bridge

to sync Ollama models into the LM Studio model directory (symlinks). You can wrap that in a script that periodically refreshes before llama-swap checks for new models:

Save as sync_and_reload.sh:

#!/bin/bash
/Users/au/src/lm-studio-ollama-bridge/lm-studio-ollama-bridge
# Force llama-swap to reload config
curl -X POST http://127.0.0.1:5005/reload-config

Cron example:

while true; do
  bash ~/sync_and_reload.sh
  sleep 3600
done


⸻

🎛 7. Custom Sampler Settings (including top_n_sigma)

llama-server supports advanced sampling flags (part of the llama.cpp sampling chain: penalties → top_n_sigma → top_k → top_p → temperature).  ￼

Example advanced sampler invocation

Replace your cmd line with:

llama-server --host 127.0.0.1 --port ${PORT} \
  --model /path/to/model.gguf \
  --top-nsigma 1.5 \
  --top-k 0 \
  --top-p 0.95 \
  --min-p 0.05 \
  --temp 1.0 \
  --typ-p 1.0

Explanation of flags:
    •    --top-nsigma: adaptive cutoff based on distribution spread
    •    --top-k: fixed count cutoff (0 = disabled)
    •    --top-p: nucleus cumulative probability cutoff
    •    --min-p: minimum token probability floor
    •    --temp: randomness scale
(Supported sampler parameters are directly passed to the llama.cpp engine.)  ￼

⸻

🔁 8. Model Folder Structure (LM Studio)

LM Studio expects models in the two-level folder structure publisher/model_name/model_file.gguf. Simple symlinks into that structure let the app see them.  ￼

Example:

~/.cache/lm-studio/models/
├── mypublisher/
│   └── mymodel/
│       └── mymodel.gguf

Your symlinker (lm-studio-ollama-bridge) ensures this structure.

⸻

🧠 Summary

You now have:

🟢 A llama-swap router serving multiple models on one port
🟢 Idle unload via TTL
🟢 Pressure-aware unloading with a small Python monitor
🟢 Shared LM Studio model paths reused by your server
🟢 Advanced sampling including top_n_sigma
🟢 A periodic bridge between Ollama and LM Studio

⸻

If you want, I can turn this into a shell script bundle / entrypoint script that manages everything (sync, start servers, start monitor) automatically for you. Just ask!
