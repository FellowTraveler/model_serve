#!/bin/bash
# Install dependencies for model serving stack

set -e

echo "=== Installing Model Serve Dependencies ==="
echo ""

# Install llama.cpp via Homebrew
if ! command -v llama-server &> /dev/null; then
    echo "Installing llama.cpp..."
    brew install llama.cpp
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

    # Determine architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_darwin_arm64.tar.gz"
    else
        TARBALL="llama-swap_${LLAMA_SWAP_VERSION}_darwin_amd64.tar.gz"
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

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Verify installation:"
echo "  llama-server --version"
echo "  llama-swap --help"
echo ""
echo "Next steps:"
echo "  1. Generate config:  ./model sync"
echo "  2. Start the stack:  ./start.sh"
echo "  3. Set up cron:      ./setup_cron.sh   (optional, for hourly auto-sync)"
echo ""
echo "Usage:"
echo "  ./model pull <name>    Pull a model"
echo "  ./model list [filter]  List models"
echo "  ./model rm <name>      Remove a model"
