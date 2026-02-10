#!/bin/bash
# 运行所有 adapter 测试脚本

echo "========================================="
echo "运行所有 Adapter 测试"
echo "========================================="

# 测试核心模块
echo ""
echo ">>> 测试核心模块..."
uv run python tests/test_core_modules.py
if [ $? -ne 0 ]; then
    echo "❌ 核心模块测试失败"
    exit 1
fi

# 测试 ASR
echo ""
echo ">>> 测试 ASR Adapter..."
uv run python tests/test_single_asr.py
if [ $? -ne 0 ]; then
    echo "⚠️  ASR 测试失败（可能缺少配置或依赖）"
fi

# 测试 TTS
echo ""
echo ">>> 测试 TTS Adapter..."
uv run python tests/test_single_tts.py
if [ $? -ne 0 ]; then
    echo "⚠️  TTS 测试失败（可能缺少配置或依赖）"
fi

# 测试 LLM
echo ""
echo ">>> 测试 LLM Adapter..."
uv run python tests/test_langchain_llm_adapter.py
if [ $? -ne 0 ]; then
    echo "⚠️  LLM 测试失败（可能缺少配置或依赖）"
fi

echo ""
echo "========================================="
echo "✅ 测试运行完成"
echo "========================================="
