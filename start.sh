#!/bin/bash
# Start the model serving stack
#
# Components:
# 1. llama-swap - Multi-model router/proxy
# 2. pressure_unloader.py - Memory pressure monitoring
# 3. sync_loop.sh - Periodic Ollama-LM Studio sync (optional)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f ".env" ]; then
    echo "Loading .env configuration..."
    set -a
    source .env
    set +a
else
    echo "Warning: .env not found, using defaults from .env.example"
    echo "Run: cp .env.example .env"
fi

LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-5847}"

# Find llama-swap (prefer system install, fallback to local bin)
find_llama_swap() {
    if command -v llama-swap &> /dev/null; then
        command -v llama-swap
    elif [ -x "$SCRIPT_DIR/bin/llama-swap" ]; then
        echo "$SCRIPT_DIR/bin/llama-swap"
    else
        echo ""
    fi
}

LLAMA_SWAP_BIN=$(find_llama_swap)

# Check dependencies
check_deps() {
    local missing=()

    if [ -z "$LLAMA_SWAP_BIN" ]; then
        missing+=("llama-swap (run ./install.sh)")
    fi

    if ! command -v llama-server &> /dev/null; then
        missing+=("llama-server (brew install llama.cpp)")
    fi

    if ! command -v python3 &> /dev/null; then
        missing+=("python3")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        echo "Error: Missing required dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi
}

# Check Python dependencies
check_python_deps() {
    python3 -c "import requests, psutil" 2>/dev/null || {
        echo "Installing Python dependencies..."
        pip3 install -r requirements.txt
    }
}

start_llama_swap() {
    echo "Starting llama-swap on port ${LLAMA_SWAP_PORT}..."
    echo "Using: $LLAMA_SWAP_BIN"
    "$LLAMA_SWAP_BIN" --config config.yaml --listen "0.0.0.0:${LLAMA_SWAP_PORT}" &
    LLAMA_SWAP_PID=$!
    echo "llama-swap PID: $LLAMA_SWAP_PID"
}

start_pressure_unloader() {
    echo "Starting pressure unloader..."
    python3 pressure_unloader.py &
    UNLOADER_PID=$!
    echo "Pressure unloader PID: $UNLOADER_PID"
}

start_sync_loop() {
    echo "Starting sync loop..."
    bash sync_loop.sh &
    SYNC_PID=$!
    echo "Sync loop PID: $SYNC_PID"
}

cleanup() {
    echo ""
    echo "Shutting down..."

    if [ -n "$LLAMA_SWAP_PID" ]; then
        kill $LLAMA_SWAP_PID 2>/dev/null || true
    fi

    if [ -n "$UNLOADER_PID" ]; then
        kill $UNLOADER_PID 2>/dev/null || true
    fi

    if [ -n "$SYNC_PID" ]; then
        kill $SYNC_PID 2>/dev/null || true
    fi

    # Kill any child processes
    pkill -P $$ 2>/dev/null || true

    echo "Done"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Main
echo "=== Model Serve Stack ==="
echo ""

check_deps
check_python_deps

# Sync models and regenerate config BEFORE starting llama-swap
echo ""
echo "Syncing models and generating config..."
./model sync

echo ""
start_llama_swap

# Wait for llama-swap to be ready
echo "Waiting for llama-swap to start..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:${LLAMA_SWAP_PORT}/running" > /dev/null 2>&1; then
        echo "llama-swap is ready!"
        break
    fi
    sleep 1
done

start_pressure_unloader
start_sync_loop

echo ""
echo "=== All services started ==="
echo "API endpoint: http://127.0.0.1:${LLAMA_SWAP_PORT}"
echo ""
echo "Endpoints:"
echo "  POST /v1/chat/completions - Chat completion (OpenAI compatible)"
echo "  GET  /running             - List loaded models"
echo "  POST /models/unload       - Unload a model"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
