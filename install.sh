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

    # Get latest release
    LLAMA_SWAP_VERSION=$(curl -s https://api.github.com/repos/mostlygeek/llama-swap/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')

    if [ -z "$LLAMA_SWAP_VERSION" ]; then
        echo "Error: Could not fetch llama-swap version"
        echo "Please install manually from: https://github.com/mostlygeek/llama-swap/releases"
        exit 1
    fi

    echo "Latest version: $LLAMA_SWAP_VERSION"

    # Determine architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        BINARY_NAME="llama-swap-darwin-arm64"
    else
        BINARY_NAME="llama-swap-darwin-amd64"
    fi

    # Download and install
    DOWNLOAD_URL="https://github.com/mostlygeek/llama-swap/releases/download/${LLAMA_SWAP_VERSION}/${BINARY_NAME}"
    echo "Downloading from: $DOWNLOAD_URL"

    curl -L -o /tmp/llama-swap "$DOWNLOAD_URL"
    chmod +x /tmp/llama-swap

    # Install to /usr/local/bin
    echo "Installing to /usr/local/bin (may require sudo)..."
    sudo mv /tmp/llama-swap /usr/local/bin/llama-swap

    echo "✓ llama-swap installed"
else
    echo "✓ llama-swap already installed"
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
echo "Start the stack:"
echo "  ./start.sh"
