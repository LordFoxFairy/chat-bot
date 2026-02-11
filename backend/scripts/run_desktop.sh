#!/bin/bash
# Development script to run Chat Bot Desktop
# 开发模式：同时启动 Python 后端和 Tauri 前端

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BACKEND_DIR")"

cd "$PROJECT_ROOT"

echo "=== Chat Bot Desktop Development ==="
echo "Project root: $PROJECT_ROOT"

# Load .env file if exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment variables from .env..."
    # Export all non-comment, non-empty lines
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        # Export the variable
        export "$line" 2>/dev/null || true
    done < "$PROJECT_ROOT/.env"
    echo "API_KEY loaded: ${API_KEY:0:20}..."
fi

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

# Start Python backend (确保从项目根目录启动)
echo "Starting Python backend..."
cd "$PROJECT_ROOT"
python3 -m backend.main server &
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
cd "$PROJECT_ROOT/frontend"

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Run Tauri dev
npm run dev

# Cleanup when Tauri exits
cleanup
