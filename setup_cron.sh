#!/bin/bash
# Manage cron-based keep-alive for model_serve
#
# Usage:
#   ./setup_cron.sh start   Install cron job and start server
#   ./setup_cron.sh stop    Remove cron job and stop server
#   ./setup_cron.sh         Show this help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/model"
LOG_FILE="$SCRIPT_DIR/model_serve.log"

# Cron entry: run every hour at minute 0
# PATH setup for cron's minimal environment (includes homebrew, etc.)
CRON_ENTRY="0 * * * * PATH=/opt/homebrew/bin:/usr/local/bin:\$HOME/.local/bin:/usr/bin:/bin:\$PATH && cd $SCRIPT_DIR && $START_SCRIPT start >> $LOG_FILE 2>&1"

show_help() {
    echo "Manage cron-based keep-alive for model_serve"
    echo ""
    echo "Usage:"
    echo "  ./setup_cron.sh start   Install cron job and start server now"
    echo "  ./setup_cron.sh stop    Remove cron job and stop server"
    echo ""
    echo "The cron job runs './model start' hourly. If the server is already"
    echo "running, it exits immediately. If not, it starts the full stack."
    echo ""
    echo "Current status:"
    if crontab -l 2>/dev/null | grep -q "model start"; then
        echo "  Cron job: INSTALLED"
    else
        echo "  Cron job: not installed"
    fi
    if pgrep -f "llama-swap.*--config" > /dev/null; then
        echo "  Server:   RUNNING"
    else
        echo "  Server:   not running"
    fi
}

do_start() {
    echo "=== Setting up model_serve keep-alive ==="
    echo ""

    # Check if cron already installed
    if crontab -l 2>/dev/null | grep -q "model start"; then
        echo "Cron job already installed."
    else
        # Add to crontab
        (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
        echo "Cron job installed (hourly keep-alive)."
    fi

    echo ""

    # Start the server now
    "$START_SCRIPT" start
}

do_stop() {
    echo "=== Removing model_serve keep-alive ==="
    echo ""

    # Remove from crontab
    if crontab -l 2>/dev/null | grep -q "model start"; then
        crontab -l | grep -v "model start" | crontab -
        echo "Cron job removed."
    else
        echo "Cron job was not installed."
    fi

    echo ""

    # Stop the server
    "$START_SCRIPT" stop
}

# Main
case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    *)
        show_help
        ;;
esac
