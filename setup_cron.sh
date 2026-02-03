#!/bin/bash
# Set up cron job to keep model_serve running
#
# This adds a cron entry to run ./model start hourly.
# If already running, it exits immediately (safe to run repeatedly).
# If not running, it starts the full stack.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/model"
LOG_FILE="$SCRIPT_DIR/model_serve.log"

# Cron entry: run every hour at minute 0
# PATH setup for cron's minimal environment (includes homebrew, etc.)
CRON_ENTRY="0 * * * * PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:/usr/bin:/bin:\$PATH && cd $SCRIPT_DIR && $START_SCRIPT start >> $LOG_FILE 2>&1"

echo "Setting up cron job for model_serve keep-alive..."
echo "Script: $START_SCRIPT start"
echo "Log: $LOG_FILE"
echo ""

# Check if already installed
if crontab -l 2>/dev/null | grep -q "model start"; then
    echo "Cron job already exists. Current crontab:"
    crontab -l | grep "model start"
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed. Will check/start server every hour."
echo ""
echo "To verify:"
echo "  crontab -l | grep 'model start'"
echo ""
echo "To remove:"
echo "  crontab -l | grep -v 'model start' | crontab -"
