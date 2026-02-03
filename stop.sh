#!/bin/bash
# Stop all model serving processes

echo "Stopping model serving stack..."

# Kill llama-swap
pkill -f "llama-swap" 2>/dev/null && echo "Stopped llama-swap" || echo "llama-swap not running"

# Kill pressure unloader
pkill -f "pressure_unloader.py" 2>/dev/null && echo "Stopped pressure unloader" || echo "Pressure unloader not running"

# Kill sync loop
pkill -f "sync_loop.sh" 2>/dev/null && echo "Stopped sync loop" || echo "Sync loop not running"

# Kill any orphaned llama-server instances
pkill -f "llama-server" 2>/dev/null && echo "Stopped llama-server instances" || echo "No llama-server instances running"

echo "Done"
