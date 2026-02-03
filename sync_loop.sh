#!/bin/bash
# Continuous sync loop for Ollama-LM Studio bridge
#
# Runs the bridge sync on a configurable interval

set -e

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

SYNC_INTERVAL="${BRIDGE_SYNC_INTERVAL:-3600}"

echo "Starting sync loop (interval: ${SYNC_INTERVAL}s)"

while true; do
    "$SCRIPT_DIR/sync_bridge.sh"
    echo "$(date): Sleeping for ${SYNC_INTERVAL}s..."
    sleep "$SYNC_INTERVAL"
done
