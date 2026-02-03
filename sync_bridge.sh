#!/bin/bash
# Sync Ollama models to LM Studio directory and regenerate config
#
# This script wraps ./model sync for use with cron.
# It sets up PATH for cron's minimal environment.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up PATH for cron (which has minimal environment)
# Include common locations for python3, homebrew, etc.
# macOS: /opt/homebrew/bin (Apple Silicon), /usr/local/bin (Intel)
# Linux: ~/.local/bin (pip --user), /home/linuxbrew/.linuxbrew/bin
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:/home/linuxbrew/.linuxbrew/bin:/usr/bin:/bin:$PATH"

# Load .env if it exists (for MODELS_DIR, etc.)
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Run the sync command
"$SCRIPT_DIR/model" sync
