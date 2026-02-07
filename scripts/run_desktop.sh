#!/bin/bash
# Development script to run Chat Bot Desktop

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Chat Bot Desktop Development ==="

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$PYTHON_PID" ]; then
        kill $PYTHON_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Python backend
echo "Starting Python backend..."
python3 app.py &
PYTHON_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "ERROR: Python backend failed to start"
    exit 1
fi

echo "Python backend started (PID: $PYTHON_PID)"

# Start Tauri development
echo "Starting Tauri development server..."
cd "$PROJECT_ROOT/desktop"

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Run Tauri dev
npm run dev

# Cleanup when Tauri exits
cleanup
