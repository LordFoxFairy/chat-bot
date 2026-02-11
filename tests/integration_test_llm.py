import asyncio
import os
import sys
from typing import List

# 添加项目根目录到 pythonpath
sys.path.append(os.getcwd())

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from backend.core.models import TextData

async def test_llm_adapter():
    """测试 LangChainLLMAdapter"""
    print("\n" + "="*50)
    print("开始测试 LangChainLLMAdapter")
    print("="*50)

    # 1. 环境变量检查
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 API_KEY 环境变量")
        return
    print(f"✅ 成功读取 API_KEY (长度: {len(api_key)})")

    # 2. 模拟配置
    config = {
        "model_name": "anthropic/claude-3.5-sonnet", # 使用 config.yaml 中的模型
        "api_key_env_var": "API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.7,
        "max_tokens": 1024
    }

    # 初始化 adapter
    try:
        adapter = LangChainLLMAdapter(module_id="test_llm", config=config)
        # 手动设置 system prompt，模拟 application context
        adapter.system_prompt = "你是一个数学助手，请只回答数字结果，不要带其他文字。"
        print(f"✅ Adapter initialized with model: {config['model_name']}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # 初始化内部 LLM
    try:
        await adapter._setup_impl()
        print("✅ LLM model client initialized")
    except Exception as e:
        print(f"❌ LLM 客户端初始化失败: {e}")
        return

    # 3. 构造输入
    question = "1+1等于几？"
    input_data = TextData(text=question)
    session_id = "test_session_001"

    print(f"\n📤 发送问题: {question}")

    # 4. 接收流式回复
    print("📥 接收回复: ", end="", flush=True)
    full_response = ""
    received_chunks = 0

    try:
        async for chunk in adapter.chat_stream(input_data, session_id):
            if chunk.is_final:
                continue

            content = chunk.text
            if content:
                print(content, end="", flush=True)
                full_response += content
                received_chunks += 1

        print("\n")

        # 5. 验证结果
        print("-" * 30)
        expected_answer = "2"
        if expected_answer in full_response:
            print(f"✅ 测试通过: 回复中包含 '{expected_answer}'")
            print(f"   完整回复: {full_response}")
            print(f"   收到 chunks: {received_chunks}")
        else:
            print(f"❌ 测试失败: 回复中未找到 '{expected_answer}'")
            print(f"   实际回复: {full_response}")

    except Exception as e:
        print(f"\n❌ 处理过程发生错误: {e}")
    finally:
        # 清理
        await adapter._close_impl()

if __name__ == "__main__":
    asyncio.run(test_llm_adapter())
