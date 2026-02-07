#!/bin/bash
# 构建 Tauri 桌面应用

# 1. 准备 Python sidecar / 可执行文件
# 需要将 python 项目打包成可执行文件 (如使用 pyinstaller)
# 假设已经打包好放在 dist/chat-bot-server

# 2. 复制到 Tauri binaries 目录
mkdir -p desktop/src-tauri/binaries/
cp dist/chat-bot-server desktop/src-tauri/binaries/python-server-x86_64-apple-darwin

# 3. 构建 Tauri 应用
cd desktop
npm install
npm run build
