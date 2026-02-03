#!/bin/bash
# Install dependencies for model serving stack

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
elif [[ "$OSTYPE" == "linux"* ]]; then
    PLATFORM="linux"
else
    echo "Unsupported platform: $OSTYPE"
    exit 1
fi

echo "=== Installing Model Serve Dependencies ==="
echo "Platform: $PLATFORM"
echo ""

# Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive
echo "✓ Submodules initialized"
echo ""

# Build lm-studio-ollama-bridge into bin/
BRIDGE_SRC="$SCRIPT_DIR/lm-studio-ollama-bridge"
BRIDGE_BIN="$SCRIPT_DIR/bin/lm-studio-ollama-bridge"

mkdir -p "$SCRIPT_DIR/bin"

if [ ! -x "$BRIDGE_BIN" ]; then
    echo "Building lm-studio-ollama-bridge..."
    if ! command -v go &> /dev/null; then
        if [ "$PLATFORM" = "macos" ]; then
            echo "Go not found. Installing via Homebrew..."
            brew install go
        else
            echo "Error: Go is required but not installed."
            echo "Install Go from: https://go.dev/dl/"
            exit 1
        fi
    fi
    cd "$BRIDGE_SRC"
    go build -o "$BRIDGE_BIN" ./cmd/ollama-sync
    cd "$SCRIPT_DIR"
    echo "✓ lm-studio-ollama-bridge built to bin/"
else
    echo "✓ lm-studio-ollama-bridge already built"
fi
echo ""

# Install llama.cpp
if ! command -v llama-server &> /dev/null; then
    if [ "$PLATFORM" = "macos" ]; then
        echo "Installing llama.cpp via Homebrew..."
        brew install llama.cpp
    else
        echo "Error: llama-server (llama.cpp) is required but not installed."
        echo "Build from source: https://github.com/ggerganov/llama.cpp"
        echo "Or install via your package manager if available."
        exit 1
    fi
else
    echo "✓ llama.cpp already installed"
fi

# Install llama-swap from GitHub releases
if ! command -v llama-swap &> /dev/null; then
    echo "Installing llama-swap..."

    # Get latest release version
    LLAMA_SWAP_VERSION=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | grep '"tag_name"' | sed -E 's/.*"v([^"]+)".*/\1/')

    if [ -z "$LLAMA_SWAP_VERSION" ]; then
        echo "Error: Could not fetch llama-swap version"
        echo "Please install manually from: https://github.com/mostlygeek/llama-swap/releases"
        exit 1
    fi

    echo "Latest version: v$LLAMA_SWAP_VERSION"

    # Determine OS and architecture
    ARCH=$(uname -m)
    if [ "$PLATFORM" = "macos" ]; then
        if [ "$ARCH" = "arm64" ]; then
            TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_darwin_arm64.tar.gz"
        else
            TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_darwin_amd64.tar.gz"
        fi
    else
        if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_linux_arm64.tar.gz"
        else
            TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz"
        fi
    fi

    # Download and extract
    DOWNLOAD_URL="https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/${TARBALL}"
    echo "Downloading from: $DOWNLOAD_URL"

    curl -L -o /tmp/llama-swap.tar.gz "$DOWNLOAD_URL"
    tar -xzf /tmp/llama-swap.tar.gz -C /tmp/

    # Install to /usr/local/bin
    echo "Installing to /usr/local/bin (requires sudo)..."
    sudo mv /tmp/llama-swap /usr/local/bin/llama-swap
    rm -f /tmp/llama-swap.tar.gz /tmp/LICENSE.md /tmp/README.md

    echo "✓ llama-swap installed to /usr/local/bin"
else
    echo "✓ llama-swap already installed at $(which llama-swap)"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -q -r requirements.txt
echo "✓ Python dependencies installed"

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "✓ Created .env - review and customize as needed"
fi

# Create MODELS_DIR if it doesn't exist (in case LM Studio isn't installed)
echo ""
echo "Ensuring models directory exists..."
source .env 2>/dev/null || true
MODELS_DIR="${MODELS_DIR:-$HOME/.cache/lm-studio/models}"
MODELS_DIR="${MODELS_DIR/#\~/$HOME}"  # Expand ~
mkdir -p "$MODELS_DIR"
echo "✓ Models directory: $MODELS_DIR"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verify installation:"
echo "  llama-server --version"
echo "  llama-swap --help"
echo ""
echo "Next steps:"
echo "  1. Generate config:  ./model sync"
echo "  2. Start the stack:  ./model start"
echo "  3. (Optional) Install as service: ./setup_service.sh install"
echo ""
echo "Usage:"
echo "  ./model pull <name>    Pull a model"
echo "  ./model list [filter]  List models"
echo "  ./model rm <name>      Remove a model"
