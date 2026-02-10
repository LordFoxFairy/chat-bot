import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from backend.core.models import TextData
from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError

@pytest.fixture
def mock_langchain_config():
    return {
        "api_key": "test_key",
        "model": "gpt-4-turbo",
        "temperature": 0.5,
        "max_history_length": 5,
        "max_retries": 1,
        "retry_delay": 0.1
    }

class AsyncIterator:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        self._iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

@pytest.mark.asyncio
async def test_initialization(mock_langchain_config):
    """测试初始化逻辑"""
    with patch("src.adapters.llm.langchain_llm_adapter.ChatOpenAI") as MockChatOpenAI:
        adapter = LangChainLLMAdapter("llm_test", mock_langchain_config)
        await adapter.setup()

        assert adapter.is_ready
        assert adapter.max_history_length == 5
        assert adapter.max_retries == 1
        MockChatOpenAI.assert_called_once()

@pytest.mark.asyncio
async def test_missing_api_key():
    """测试缺少 API Key 抛出异常"""
    with pytest.raises(ModuleInitializationError):
        LangChainLLMAdapter("llm_test", {"model": "gpt-4"})

@pytest.mark.asyncio
async def test_chat_stream(mock_langchain_config):
    """测试流式对话及历史记录功能"""
    with patch("src.adapters.llm.langchain_llm_adapter.ChatOpenAI") as MockChatOpenAI:
        # Mock LLM 响应
        mock_llm = MagicMock()
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.content = " World"

        mock_llm.astream.return_value = AsyncIterator([mock_chunk1, mock_chunk2])
        MockChatOpenAI.return_value = mock_llm

        adapter = LangChainLLMAdapter("llm_test", mock_langchain_config)
        await adapter.setup()

        # 第一次对话
        text_input = TextData(text="Hi")
        chunks = []
        async for chunk in adapter.chat_stream(text_input, "session_1"):
            if not chunk.is_final:
                chunks.append(chunk.text)

        assert "".join(chunks) == "Hello World"
        assert len(adapter.chat_histories["session_1"]) == 3  # System + User + AI

        # 验证历史记录修剪
        # 模拟填充历史记录超过限制 (5)
        from langchain_core.messages import HumanMessage, AIMessage
        adapter.chat_histories["session_1"].extend([
            HumanMessage(content="1"), AIMessage(content="1"),
            HumanMessage(content="2"), AIMessage(content="2"),
        ])

        # 再次对话触发修剪
        mock_llm.astream.return_value = AsyncIterator([mock_chunk1])
        async for _ in adapter.chat_stream(TextData(text="Trim test"), "session_1"):
            pass

        assert len(adapter.chat_histories["session_1"]) <= 5 + 1  # 5 + 新回复

@pytest.mark.asyncio
async def test_retry_mechanism(mock_langchain_config):
    """测试重试机制"""
    with patch("src.adapters.llm.langchain_llm_adapter.ChatOpenAI") as MockChatOpenAI:
        mock_llm = MagicMock()
        # 第一次抛出异常，第二次成功
        mock_chunk = MagicMock()
        mock_chunk.content = "Success"

        # 模拟 astream 第一次失败，第二次成功
        # 注意: 这里比较 trick，我们需要 mock astream 每次调用返回不同的 iterator

        async def fail_then_succeed(*args, **kwargs):
            if not hasattr(fail_then_succeed, 'called'):
                fail_then_succeed.called = True
                raise ConnectionError("Network error")
            yield mock_chunk

        mock_llm.astream.side_effect = fail_then_succeed
        MockChatOpenAI.return_value = mock_llm

        adapter = LangChainLLMAdapter("llm_test", mock_langchain_config)
        await adapter.setup()

        chunks = []
        async for chunk in adapter.chat_stream(TextData(text="Retry test"), "session_retry"):
            if not chunk.is_final:
                chunks.append(chunk.text)

        assert "Success" in chunks

@pytest.mark.asyncio
async def test_clear_session_history(mock_langchain_config):
    """测试手动清理会话历史"""
    with patch("src.adapters.llm.langchain_llm_adapter.ChatOpenAI"):
        adapter = LangChainLLMAdapter("llm_test", mock_langchain_config)
        adapter.chat_histories["session_x"] = []

        adapter.clear_session_history("session_x")
        assert "session_x" not in adapter.chat_histories
