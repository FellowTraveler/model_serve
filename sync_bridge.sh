#!/bin/bash
# Sync Ollama models to LM Studio directory and regenerate config
#
# This script:
# 1. Cleans up broken symlinks (from deleted Ollama models)
# 2. Runs the Ollama-LM Studio bridge to sync symlinks
# 3. Regenerates config.yaml with all discovered models
# 4. Tells llama-swap to reload its configuration

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Expand ~ in paths
BRIDGE_SCRIPT="${BRIDGE_SCRIPT/#\~/$HOME}"
LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-5847}"
API_URL="http://127.0.0.1:${LLAMA_SWAP_PORT}"

# Default to bin/ if not set
if [ -z "$BRIDGE_SCRIPT" ]; then
    BRIDGE_SCRIPT="$SCRIPT_DIR/bin/lm-studio-ollama-bridge"
fi

MODELS_DIR="${MODELS_DIR:-$HOME/.cache/lm-studio/models}"
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"

# Step 1: Clean up broken symlinks (from deleted Ollama models)
echo "$(date): Cleaning up broken symlinks..."
broken_count=$(find "$MODELS_DIR" -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l | tr -d ' ')
if [ "$broken_count" -gt 0 ]; then
    find "$MODELS_DIR" -type l ! -exec test -e {} \; -delete
    find "$MODELS_DIR" -type d -empty -delete 2>/dev/null || true
    echo "$(date): Removed $broken_count broken symlinks"
else
    echo "$(date): No broken symlinks found"
fi

# Step 2: Run bridge sync
echo "$(date): Running Ollama-LM Studio bridge sync..."

if [ -x "$BRIDGE_SCRIPT" ]; then
    "$BRIDGE_SCRIPT"
    echo "$(date): Bridge sync completed"
else
    echo "$(date): Warning - Bridge not found: $BRIDGE_SCRIPT"
    echo "$(date): Run ./install.sh to build it"
fi

# Step 3: Regenerate config
echo "$(date): Regenerating config.yaml..."
python3 "$SCRIPT_DIR/generate_config.py"
echo "$(date): Config regenerated"

# Step 4: Reload llama-swap config if it's running
if curl -s "${API_URL}/running" > /dev/null 2>&1; then
    echo "$(date): Reloading llama-swap configuration..."
    curl -s -X POST "${API_URL}/reload-config" || echo "$(date): Warning - Failed to reload config"
    echo "$(date): Config reload requested"
else
    echo "$(date): llama-swap not running, skipping config reload"
fi
