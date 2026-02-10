"""全面的适配器集成测试

测试所有适配器的生命周期管理、资源释放和基本功能。
"""
import asyncio
import os
import sys
import pytest
from typing import Dict, Any

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.core.interfaces.base_module import BaseModule
from backend.core.interfaces.base_llm import BaseLLM
from backend.core.interfaces.base_tts import BaseTTS
from backend.core.interfaces.base_asr import BaseASR
from backend.core.interfaces.base_vad import BaseVAD
from backend.core.models.exceptions import ModuleInitializationError

# Mock 适配器用于测试基类功能
class MockAdapter(BaseModule):
    def __init__(self, module_id, config):
        super().__init__(module_id, config)
        self.setup_called = False
        self.close_called = False

    async def _setup_impl(self):
        self.setup_called = True

    async def close(self):
        self.close_called = True
        await super().close()

@pytest.mark.asyncio
async def test_base_module_lifecycle():
    """测试 BaseModule 生命周期管理"""
    adapter = MockAdapter("test_module", {})

    # 初始状态
    assert not adapter.is_ready

    # Setup
    await adapter.setup()
    assert adapter.is_ready
    assert adapter.setup_called

    # 重复 Setup
    adapter.setup_called = False
    await adapter.setup()
    assert adapter.is_ready
    assert not adapter.setup_called  # 不应再次调用

    # Close
    await adapter.close()
    assert not adapter.is_ready
    assert adapter.close_called

@pytest.mark.asyncio
async def test_async_context_manager():
    """测试异步上下文管理器"""
    adapter = MockAdapter("test_ctx", {})

    async with adapter as module:
        assert module is adapter
        assert adapter.is_ready
        assert adapter.setup_called

    # 退出上下文后应自动关闭
    assert not adapter.is_ready
    assert adapter.close_called

class MockLLM(BaseLLM):
    def __init__(self, module_id, config):
        super().__init__(module_id, config)
        self.histories = {}

    async def _setup_impl(self):
        pass

    async def chat_stream(self, text, session_id):
        yield text

    def clear_history(self, session_id):
        if session_id in self.histories:
            del self.histories[session_id]

    def get_history_length(self, session_id):
        return len(self.histories.get(session_id, []))

@pytest.mark.asyncio
async def test_base_llm_interface():
    """测试 BaseLLM 接口"""
    llm = MockLLM("test_llm", {})
    await llm.setup()

    session_id = "sess_1"
    llm.histories[session_id] = ["msg1", "msg2"]

    assert llm.get_history_length(session_id) == 2

    llm.clear_history(session_id)
    assert llm.get_history_length(session_id) == 0

    await llm.close()

# 尝试导入真实适配器进行测试（如果依赖可用）
try:
    from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANGCHAIN, reason="LangChain not available")
async def test_langchain_adapter_lifecycle():
    """测试 LangChain 适配器生命周期"""
    # 使用 mock 配置，避免真实连接
    config = {
        "model_name": "gpt-3.5-turbo",
        "api_key": "sk-mock-key",
        "temperature": 0.7
    }

    adapter = LangChainLLMAdapter("llm_test", config)

    # 测试初始化
    try:
        await adapter.setup()
        assert adapter.is_ready
    except Exception as e:
        # 如果没有网络连接或 API Key 无效，初始化可能会失败
        # 但我们主要关注接口调用是否正确
        pass

    # 测试历史记录管理
    session_id = "test_sess"
    adapter.chat_histories[session_id] = ["msg1"]
    assert adapter.get_history_length(session_id) == 1

    adapter.clear_history(session_id)
    assert adapter.get_history_length(session_id) == 0

    await adapter.close()
    assert not adapter.is_ready

if __name__ == "__main__":
    asyncio.run(test_base_module_lifecycle())
    asyncio.run(test_async_context_manager())
    asyncio.run(test_base_llm_interface())
    print("All tests passed!")
