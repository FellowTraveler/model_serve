#!/bin/bash
# Check status of model serving stack

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-5847}"
API_URL="http://127.0.0.1:${LLAMA_SWAP_PORT}"

echo "=== Model Serve Status ==="
echo ""

# Check llama-swap
if pgrep -f "llama-swap" > /dev/null; then
    echo "✓ llama-swap: running"
else
    echo "✗ llama-swap: not running"
fi

# Check Ollama
OLLAMA_BASE="${OLLAMA_BASE:-http://localhost:11434}"
if curl -s "${OLLAMA_BASE}/api/version" > /dev/null 2>&1; then
    OLLAMA_VERSION=$(curl -s "${OLLAMA_BASE}/api/version" | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','?'))" 2>/dev/null || echo "?")
    echo "✓ ollama: running (v${OLLAMA_VERSION} at ${OLLAMA_BASE})"
else
    echo "✗ ollama: not running (expected at ${OLLAMA_BASE})"
fi

# Check LM Studio
LM_STUDIO_BASE="${LM_STUDIO_BASE:-http://127.0.0.1:1234}"
if curl -s "${LM_STUDIO_BASE}/v1/models" > /dev/null 2>&1; then
    echo "✓ lm_studio: running (at ${LM_STUDIO_BASE})"
else
    echo "- lm_studio: not running (expected at ${LM_STUDIO_BASE})"
fi

# Check pressure unloader
if pgrep -f "pressure_unloader.py" > /dev/null; then
    echo "✓ pressure_unloader: running"
else
    echo "✗ pressure_unloader: not running"
fi

# Check sync loop
if pgrep -f "sync_loop.sh" > /dev/null; then
    echo "✓ sync_loop: running"
else
    echo "✗ sync_loop: not running"
fi

echo ""

# Check API and loaded models
if curl -s "${API_URL}/running" > /dev/null 2>&1; then
    echo "API endpoint: ${API_URL}"
    echo ""
    echo "Loaded models:"
    curl -s "${API_URL}/running" | python3 -m json.tool 2>/dev/null || curl -s "${API_URL}/running"
else
    echo "API not responding at ${API_URL}"
fi

echo ""

# Memory status
echo "System memory:"
python3 -c "import psutil; m=psutil.virtual_memory(); print(f'  Used: {m.percent}% ({m.used/1024**3:.1f}GB / {m.total/1024**3:.1f}GB)')" 2>/dev/null || echo "  (install psutil for memory info)"
