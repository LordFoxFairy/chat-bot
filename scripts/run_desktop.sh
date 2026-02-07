#!/bin/bash
# 启动 Tauri 开发环境
# 先启动 Python 后端，再启动 Tauri 前端

# 确保 desktop/src-tauri/binaries/python-server 存在或指向正确位置
# 在开发模式下，我们可以并行运行 python main.py 和 tauri dev

# 启动 Python 后端 (假设在根目录运行)
echo "Starting Python backend..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 app.py &
BACKEND_PID=$!

# 等待后端启动
sleep 2

# 启动 Tauri
echo "Starting Tauri app..."
cd desktop
npm install
# npm run dev
tauri dev

# 清理
kill $BACKEND_PID
