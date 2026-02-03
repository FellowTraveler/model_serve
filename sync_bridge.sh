#!/bin/bash
# Sync Ollama models to LM Studio directory and regenerate config
#
# This script:
# 1. Runs the Ollama-LM Studio bridge to sync symlinks
# 2. Regenerates config.yaml with all discovered models
# 3. Tells llama-swap to reload its configuration

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

BRIDGE_SCRIPT="${BRIDGE_SCRIPT:-/Users/au/src/lm-studio-ollama-bridge/lm-studio-ollama-bridge}"
LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-5847}"
API_URL="http://127.0.0.1:${LLAMA_SWAP_PORT}"

# Step 1: Run bridge sync
echo "$(date): Running Ollama-LM Studio bridge sync..."

if [ -x "$BRIDGE_SCRIPT" ]; then
    "$BRIDGE_SCRIPT"
    echo "$(date): Bridge sync completed"
else
    echo "$(date): Warning - Bridge script not found or not executable: $BRIDGE_SCRIPT"
fi

# Step 2: Regenerate config
echo "$(date): Regenerating config.yaml..."
python3 "$SCRIPT_DIR/generate_config.py"
echo "$(date): Config regenerated"

# Step 3: Reload llama-swap config if it's running
if curl -s "${API_URL}/running" > /dev/null 2>&1; then
    echo "$(date): Reloading llama-swap configuration..."
    curl -s -X POST "${API_URL}/reload-config" || echo "$(date): Warning - Failed to reload config"
    echo "$(date): Config reload requested"
else
    echo "$(date): llama-swap not running, skipping config reload"
fi
