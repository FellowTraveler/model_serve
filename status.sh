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
