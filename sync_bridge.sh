#!/bin/bash
# Sync Ollama models to LM Studio directory and regenerate config
#
# This script wraps ./model sync for use with cron.
# All sync logic is in the model script.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the sync command
"$SCRIPT_DIR/model" sync
