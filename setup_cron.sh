#!/bin/bash
# Set up cron job for periodic model sync
#
# This adds a cron entry to run sync_bridge.sh hourly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_bridge.sh"
LOG_FILE="$SCRIPT_DIR/sync.log"

# Cron entry: run every hour at minute 0
CRON_ENTRY="0 * * * * $SYNC_SCRIPT >> $LOG_FILE 2>&1"

echo "Setting up cron job for model sync..."
echo "Script: $SYNC_SCRIPT"
echo "Log: $LOG_FILE"
echo ""

# Check if already installed
if crontab -l 2>/dev/null | grep -q "$SYNC_SCRIPT"; then
    echo "Cron job already exists. Current crontab:"
    crontab -l | grep sync_bridge
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed. Sync will run every hour."
echo ""
echo "To verify:"
echo "  crontab -l | grep sync_bridge"
echo ""
echo "To remove:"
echo "  crontab -l | grep -v sync_bridge | crontab -"
