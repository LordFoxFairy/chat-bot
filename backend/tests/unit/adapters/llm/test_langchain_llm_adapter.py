import os
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from typing import AsyncGenerator

from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from backend.core.models import TextData
from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

@pytest.fixture
def mock_config():
    return {
        "model_name": "gpt-3.5-turbo",
        "api_key": "test-api-key",
        "temperature": 0.7,
        "max_tokens": 100,
        "system_prompt": "You are a test bot."
    }

class TestLangChainLLMAdapterInitialization:

    def test_init_with_config_api_key(self, mock_config):
        """测试从配置中读取 API Key"""
        # 清除环境变量，确保使用配置中的 api_key
        with patch.dict(os.environ, {}, clear=True):
            adapter = LangChainLLMAdapter("llm_test", mock_config)
            assert adapter.api_key == "test-api-key"
            assert adapter.model_name == "gpt-3.5-turbo"

    def test_init_with_env_api_key(self, mock_config):
        """测试优先使用环境变量中的 API Key"""
        mock_config_no_key = mock_config.copy()
        del mock_config_no_key["api_key"]

        with patch.dict(os.environ, {"API_KEY": "env-api-key"}):
            adapter = LangChainLLMAdapter("llm_test", mock_config_no_key)
            assert adapter.api_key == "env-api-key"

        # 测试自定义环境变量名
        mock_config_custom_env = mock_config_no_key.copy()
        mock_config_custom_env["api_key_env_var"] = "CUSTOM_KEY"

        with patch.dict(os.environ, {"CUSTOM_KEY": "custom-env-key"}):
            adapter = LangChainLLMAdapter("llm_test", mock_config_custom_env)
            assert adapter.api_key == "custom-env-key"

    def test_init_missing_api_key(self, mock_config):
        """测试缺少 API Key 时抛出异常"""
        mock_config_no_key = mock_config.copy()
        del mock_config_no_key["api_key"]

        # 确保环境变量未设置
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ModuleInitializationError, match="缺少 API Key"):
                LangChainLLMAdapter("llm_test", mock_config_no_key)

    def test_resolve_base_url(self, mock_config):
        """测试 Base URL 解析"""
        # 测试从配置读取
        config_with_url = mock_config.copy()
        config_with_url["base_url"] = "https://api.example.com"
        adapter = LangChainLLMAdapter("llm_test", config_with_url)
        assert adapter.base_url == "https://api.example.com"

        # 测试从环境变量读取
        config_with_env = mock_config.copy()
        config_with_env["base_url_env_var"] = "LLM_BASE_URL"

        with patch.dict(os.environ, {"LLM_BASE_URL": "https://env.example.com"}):
            adapter = LangChainLLMAdapter("llm_test", config_with_env)
            assert adapter.base_url == "https://env.example.com"

    def test_init_model_auto(self, mock_config):
        """测试使用 init_chat_model 自动初始化模型"""
        # 使用 patch.dict 注入 mock module 到 sys.modules
        # 我们需要 mock 'langchain.chat_models' 模块

        mock_chat_models = MagicMock()
        mock_init = MagicMock()
        mock_chat_models.init_chat_model = mock_init

        # 清除环境变量，确保使用配置中的 api_key
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict("sys.modules", {"langchain.chat_models": mock_chat_models}):
                adapter = LangChainLLMAdapter("llm_test", mock_config)

                # 手动触发 _init_model
                adapter._init_model()

                mock_init.assert_called_once()
                call_kwargs = mock_init.call_args[1]
                assert call_kwargs["model"] == "gpt-3.5-turbo"
                assert call_kwargs["api_key"] == "test-api-key"
                assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_setup_success(self, mock_config):
        """测试 setup 成功"""
        adapter = LangChainLLMAdapter("llm_test", mock_config)

        # Mock _init_model 因为它包含外部依赖调用
        with patch.object(adapter, '_init_model') as mock_init:
            mock_llm = MagicMock()
            mock_init.return_value = mock_llm

            # 修正: 调用基类的 start 方法需要它是 async 的且存在
            # 检查 BaseModule，如果是同步的 start，或者是 _setup_impl
            # 在 src/core/interfaces/base_module.py 中 start 应该是 async并调用 _setup_impl
            # 如果没有看到 BaseModule 源码，我们假设它是遵循标准模式
            # 这里直接调用 _setup_impl 来测试 adapter 的初始化逻辑
            await adapter._setup_impl()

            assert adapter.llm is mock_llm
            # is_ready 通常由 base module 管理，这里我们只测 _setup_impl 逻辑

@pytest.mark.asyncio
class TestLangChainLLMChat:

    @pytest.fixture
    async def adapter(self, mock_config):
        adapter = LangChainLLMAdapter("llm_test", mock_config)
        # 手动设置 llm 避免调用真实初始化
        adapter.llm = AsyncMock()
        adapter.is_active = True # 模拟已启动
        return adapter

    async def test_chat_stream_basic(self, adapter):
        """测试基本流式对话"""
        # 模拟 LLM 响应
        mock_chunk1 = MagicMock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = MagicMock()
        mock_chunk2.content = " World"

        async def mock_astream(*args, **kwargs):
            yield mock_chunk1
            yield mock_chunk2

        adapter.llm.astream = mock_astream

        input_text = TextData(text="Hi", is_final=True)
        session_id = "sess_1"

        chunks = []
        async for chunk in adapter.chat_stream(input_text, session_id):
            chunks.append(chunk)

        assert len(chunks) == 3 # Hello, World, End(empty)
        assert chunks[0].text == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].text == " World"
        assert chunks[2].text == ""
        assert chunks[2].is_final is True

        # 验证历史记录
        history = adapter.chat_histories[session_id]
        assert len(history) == 3 # System, Human, AI
        assert isinstance(history[0], SystemMessage)
        assert isinstance(history[1], HumanMessage)
        assert isinstance(history[1], HumanMessage) and history[1].content == "Hi"
        # 注意：源码中是在流结束后添加到历史
        assert isinstance(history[2], AIMessage)
        assert history[2].content == "Hello World"

    async def test_chat_stream_history_trimming(self, adapter):
        """测试历史记录修剪"""
        adapter.max_history_length = 3
        session_id = "sess_hist"

        adapter.chat_histories[session_id] = [
            SystemMessage(content="Sys"),
            HumanMessage(content="H1"),
            AIMessage(content="A1"),
            HumanMessage(content="H2"),
            AIMessage(content="A2")
        ]

        # 修正: astream 是一个异步生成器方法。
        # 当被调用时，它返回一个异步迭代器。
        # 我们的 Mock 需要表现得像一个函数，该函数调用时返回一个异步迭代器。

        # 方法 1: 使用 MagicMock 并手动实现 __aiter__
        # 方法 2: 定义一个 async generator function 并赋值给 side_effect

        async def mock_astream(*args, **kwargs):
            yield MagicMock(content="A3")

        adapter.llm.astream = mock_astream  # 直接赋值函数

        input_text = TextData(text="H3", is_final=True)

        async for _ in adapter.chat_stream(input_text, session_id):
            pass

        history = adapter.chat_histories[session_id]
        assert len(history) <= adapter.max_history_length + 1
        assert isinstance(history[0], SystemMessage)
        assert history[-1].content == "A3"
        assert history[-2].content == "H3"

    async def test_session_isolation(self, adapter):
        """测试多会话隔离"""

        async def gen_resp1(*args, **kwargs):
            yield MagicMock(content="Resp1")

        async def gen_resp2(*args, **kwargs):
            yield MagicMock(content="Resp2")

        # 使用 side_effect 返回不同的 generator
        # adapter.llm.astream 必须是一个 callable
        adapter.llm.astream = MagicMock(side_effect=[gen_resp1(), gen_resp2()])

        # chat_stream 调用 adapter.llm.astream(...)
        # 第一次调用返回 gen_resp1() 的结果 (async generator)
        # 第二次调用返回 gen_resp2() 的结果

        async for _ in adapter.chat_stream(TextData(text="Hi1"), "s1"): pass
        async for _ in adapter.chat_stream(TextData(text="Hi2"), "s2"): pass

        h1 = adapter.chat_histories["s1"]
        h2 = adapter.chat_histories["s2"]

        assert len(h1) == 3
        assert h1[1].content == "Hi1"
        assert h1[2].content == "Resp1"

        assert len(h2) == 3
        assert h2[1].content == "Hi2"
        assert h2[2].content == "Resp2"

    async def test_error_handling_and_retry(self, adapter):
        """测试错误重试"""
        adapter.retry_delay = 0.01

        # 模拟前两次失败，第三次成功
        mock_chunk = MagicMock()
        mock_chunk.content = "Success"

        # 创建一个可迭代的副作用
        # 第一次调用 astream 返回一个生成器，该生成器抛出错误
        # 第二次同上
        # 第三次返回正常生成器

        async def fail_gen(*args, **kwargs):
            raise Exception("Fail")
            yield # never reached

        async def success_gen(*args, **kwargs):
            yield mock_chunk

        # 设置 side_effect
        # 注意 astream 是 async generator function 调用后返回 async generator
        adapter.llm.astream = MagicMock(side_effect=[
            fail_gen(),
            fail_gen(),
            success_gen()
        ])

        chunks = []
        async for chunk in adapter.chat_stream(TextData(text="Test"), "retry_sess"):
            chunks.append(chunk)

        # 验证结果
        assert any(c.text == "Success" for c in chunks)
        # 验证调用次数: 1 initial + 2 retries = 3 calls
        assert adapter.llm.astream.call_count == 3

    async def test_empty_input(self, adapter):
        """测试空输入"""
        chunks = []
        # 使用 is_final=True 允许空文本（结束信号），或者提供非空文本但会被视为 invalid 如果逻辑是这样
        # 源码 check: if not text.text or not text.text.strip(): yield empty final
        # 但 TextData 校验要求非 final 时 text 不为空
        # 所以我们测试 "   " (space) for non-final?
        # TextData validate_text_content allows spaces IF is_final=True. Or fails if not.
        # So input must be valid TextData.

        # Scenario 1: Empty text with is_final=True
        input_text = TextData(text="", is_final=True)
        async for chunk in adapter.chat_stream(input_text, "sess_empty"):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].is_final is True
        assert chunks[0].text == ""
        # 确保未调用 LLM
        adapter.llm.astream.assert_not_called()

        # Scenario 2: Spaces (valid only if is_final=True given TextData validation)
        # If we pass is_final=False, TextData raises error for empty/spaces.
        # So adapter only sees valid TextData.


    @pytest.mark.asyncio
    async def test_history_management(self, adapter):
        """测试 clear_history 和 get_history_length"""
        adapter.chat_histories["s_test"] = [SystemMessage(content="s")]

        assert adapter.get_history_length("s_test") == 1
        assert adapter.get_history_length("non_exist") == 0

        adapter.clear_history("s_test")
        assert adapter.get_history_length("s_test") == 0
