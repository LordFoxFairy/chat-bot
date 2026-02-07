#!/bin/bash
# Build script for Chat Bot Desktop production build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Chat Bot Desktop Production Build ==="

# Step 1: Build Python executable
echo ""
echo "Step 1: Building Python executable..."

# Install build dependencies if needed
pip install pyinstaller -q

# Build Python executable
echo "Running PyInstaller..."
pyinstaller \
    --onefile \
    --name python-server \
    --distpath dist \
    --workpath build \
    --clean \
    --noconfirm \
    app.py

echo "Python executable built: dist/python-server"

# Step 2: Copy Python executable to Tauri binaries
echo ""
echo "Step 2: Preparing Tauri binaries..."

BINARIES_DIR="$PROJECT_ROOT/desktop/src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

# Detect platform
case "$(uname -s)" in
    Darwin)
        case "$(uname -m)" in
            arm64)
                TARGET="aarch64-apple-darwin"
                ;;
            x86_64)
                TARGET="x86_64-apple-darwin"
                ;;
        esac
        ;;
    Linux)
        case "$(uname -m)" in
            x86_64)
                TARGET="x86_64-unknown-linux-gnu"
                ;;
            aarch64)
                TARGET="aarch64-unknown-linux-gnu"
                ;;
        esac
        ;;
    MINGW*|MSYS*|CYGWIN*)
        TARGET="x86_64-pc-windows-msvc"
        ;;
esac

if [ -z "$TARGET" ]; then
    echo "ERROR: Unsupported platform"
    exit 1
fi

echo "Target platform: $TARGET"

# Copy with platform-specific name
cp "dist/python-server" "$BINARIES_DIR/python-server-$TARGET"
chmod +x "$BINARIES_DIR/python-server-$TARGET"

echo "Binary copied to: $BINARIES_DIR/python-server-$TARGET"

# Step 3: Build Tauri application
echo ""
echo "Step 3: Building Tauri application..."

cd "$PROJECT_ROOT/desktop"

# Install npm dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Build Tauri
npm run build

echo ""
echo "=== Build Complete ==="
echo "Output: desktop/src-tauri/target/release/bundle/"
