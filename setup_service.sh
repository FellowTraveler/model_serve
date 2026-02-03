#!/bin/bash
# Install model_serve as a system service
#
# Usage:
#   ./setup_service.sh install    Install and start service
#   ./setup_service.sh uninstall  Stop and remove service
#   ./setup_service.sh start      Start the service
#   ./setup_service.sh stop       Stop the service
#   ./setup_service.sh status     Check service status
#   ./setup_service.sh            Show this help
#
# Supports:
#   - macOS: launchd (LaunchAgent)
#   - Linux: systemd (user service)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="com.model-serve.agent"
LOG_FILE="$SCRIPT_DIR/model_serve.log"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    PLIST_PATH="$HOME/Library/LaunchAgents/${SERVICE_NAME}.plist"
elif [[ "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
    SYSTEMD_DIR="$HOME/.config/systemd/user"
    SERVICE_FILE="$SYSTEMD_DIR/model-serve.service"
else
    echo "Unsupported platform: $OSTYPE"
    echo "Use ./setup_cron.sh as a fallback."
    exit 1
fi

show_help() {
    echo "Install model_serve as a system service"
    echo ""
    echo "Usage:"
    echo "  ./setup_service.sh install    Install and start service"
    echo "  ./setup_service.sh uninstall  Stop and remove service"
    echo "  ./setup_service.sh start      Start the service"
    echo "  ./setup_service.sh stop       Stop the service"
    echo "  ./setup_service.sh status     Check service status"
    echo ""
    echo "Platform: $PLATFORM"
    echo ""
    show_status
}

# ============================================================================
# macOS (launchd)
# ============================================================================

macos_install() {
    echo "Installing launchd service..."

    # Create plist
    cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${SCRIPT_DIR}/start.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>${LOG_FILE}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

    echo "Created: $PLIST_PATH"

    # Load the service
    launchctl load "$PLIST_PATH"
    echo "Service installed and started."
    echo "Logs: $LOG_FILE"
}

macos_uninstall() {
    echo "Removing launchd service..."

    if [ -f "$PLIST_PATH" ]; then
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        rm "$PLIST_PATH"
        echo "Service removed."
    else
        echo "Service was not installed."
    fi

    # Also stop any running processes
    "$SCRIPT_DIR/model" stop 2>/dev/null || true
}

macos_start() {
    if [ -f "$PLIST_PATH" ]; then
        launchctl load "$PLIST_PATH" 2>/dev/null || launchctl start "$SERVICE_NAME"
        echo "Service started."
    else
        echo "Service not installed. Run './setup_service.sh install' first."
        exit 1
    fi
}

macos_stop() {
    if [ -f "$PLIST_PATH" ]; then
        launchctl stop "$SERVICE_NAME" 2>/dev/null || true
        echo "Service stopped."
    else
        "$SCRIPT_DIR/model" stop
    fi
}

macos_status() {
    if [ -f "$PLIST_PATH" ]; then
        echo "  Service: INSTALLED"
        if launchctl list | grep -q "$SERVICE_NAME"; then
            echo "  launchd: LOADED"
        else
            echo "  launchd: not loaded"
        fi
    else
        echo "  Service: not installed"
    fi
}

# ============================================================================
# Linux (systemd)
# ============================================================================

linux_install() {
    echo "Installing systemd user service..."

    # Create systemd user directory if needed
    mkdir -p "$SYSTEMD_DIR"

    # Create service file
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Model Serve - Multi-model LLM inference server
After=network.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/start.sh
Restart=always
RestartSec=10
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}
Environment=PATH=/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
EOF

    echo "Created: $SERVICE_FILE"

    # Reload and enable
    systemctl --user daemon-reload
    systemctl --user enable model-serve
    systemctl --user start model-serve

    # Enable lingering so service runs even when logged out
    loginctl enable-linger "$USER" 2>/dev/null || true

    echo "Service installed and started."
    echo "Logs: $LOG_FILE"
}

linux_uninstall() {
    echo "Removing systemd user service..."

    if [ -f "$SERVICE_FILE" ]; then
        systemctl --user stop model-serve 2>/dev/null || true
        systemctl --user disable model-serve 2>/dev/null || true
        rm "$SERVICE_FILE"
        systemctl --user daemon-reload
        echo "Service removed."
    else
        echo "Service was not installed."
    fi

    # Also stop any running processes
    "$SCRIPT_DIR/model" stop 2>/dev/null || true
}

linux_start() {
    if [ -f "$SERVICE_FILE" ]; then
        systemctl --user start model-serve
        echo "Service started."
    else
        echo "Service not installed. Run './setup_service.sh install' first."
        exit 1
    fi
}

linux_stop() {
    if [ -f "$SERVICE_FILE" ]; then
        systemctl --user stop model-serve
        echo "Service stopped."
    else
        "$SCRIPT_DIR/model" stop
    fi
}

linux_status() {
    if [ -f "$SERVICE_FILE" ]; then
        echo "  Service: INSTALLED"
        if systemctl --user is-active model-serve >/dev/null 2>&1; then
            echo "  systemd: RUNNING"
        else
            echo "  systemd: stopped"
        fi
    else
        echo "  Service: not installed"
    fi
}

# ============================================================================
# Common
# ============================================================================

show_status() {
    echo "Current status:"
    if [ "$PLATFORM" = "macos" ]; then
        macos_status
    else
        linux_status
    fi

    if pgrep -f "llama-swap.*--config" > /dev/null; then
        echo "  Server:  RUNNING"
    else
        echo "  Server:  not running"
    fi
}

do_install() {
    if [ "$PLATFORM" = "macos" ]; then
        macos_install
    else
        linux_install
    fi
}

do_uninstall() {
    if [ "$PLATFORM" = "macos" ]; then
        macos_uninstall
    else
        linux_uninstall
    fi
}

do_start() {
    if [ "$PLATFORM" = "macos" ]; then
        macos_start
    else
        linux_start
    fi
}

do_stop() {
    if [ "$PLATFORM" = "macos" ]; then
        macos_stop
    else
        linux_stop
    fi
}

# Main
case "${1:-}" in
    install)
        do_install
        ;;
    uninstall)
        do_uninstall
        ;;
    start)
        do_start
        ;;
    stop)
        do_stop
        ;;
    status)
        show_status
        ;;
    *)
        show_help
        ;;
esac
