"""ChatEngine 集成测试"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.core.engine.chat_engine import ChatEngine
from backend.core.session.session_manager import SessionManager, InMemoryStorage
from backend.core.app_context import AppContext
from backend.core.adapter_loader import AdapterLoader
from backend.core.interfaces.base_module import BaseModule


class MockModule(BaseModule):
    """用于测试的 Mock 模块"""
    async def _setup_impl(self):
        pass

    async def _close_impl(self):
        pass


@pytest.mark.asyncio
class TestChatEngineIntegration:
    """ChatEngine 集成测试类"""

    def setup_method(self):
        """每个测试前初始化"""
        AppContext.clear()
        self.storage = InMemoryStorage()
        self.session_manager = SessionManager(storage_backend=self.storage)

    def teardown_method(self):
        """每个测试后清空"""
        AppContext.clear()

    async def test_chat_engine_initialization(self):
        """测试 ChatEngine 初始化"""
        config = {
            "modules": {}  # 空配置
        }

        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager
        )

        # 验证基本属性
        assert engine.global_config == config
        assert engine.session_manager is self.session_manager
        assert engine.conversation_manager is not None
        assert isinstance(engine.common_modules, dict)
        assert isinstance(engine.protocol_modules, dict)

    async def test_initialize_with_empty_config(self):
        """测试空配置初始化"""
        config = {"modules": {}}

        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager
        )

        await engine.initialize()

        # AppContext 应该被设置（即使模块为空）
        assert AppContext.get_module("llm") is None  # 无模块

    async def test_initialize_with_mock_modules(self):
        """测试带模块的初始化（模拟）"""
        config = {
            "modules": {
                "llm": {
                    "adapter_type": "mock_llm",
                    "config": {}
                }
            }
        }

        # 准备 Mock Loader 和 模块
        loader = AdapterLoader()
        mock_llm = MockModule("llm", {})

        # 注册 Mock 工厂
        mock_factory = Mock(return_value=mock_llm)
        loader.register("llm", mock_factory)

        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager,
            adapter_loader=loader
        )

        await engine.initialize()

        # 验证 LLM 被创建
        mock_factory.assert_called_once()

        # 验证模块在 common_modules 中
        assert "llm" in engine.common_modules

        # 验证 AppContext 已设置
        assert AppContext.get_module("llm") == mock_llm

    async def test_get_module(self):
        """测试获取模块"""
        config = {"modules": {}}
        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager
        )

        # 添加模拟模块
        mock_module = Mock()
        engine.common_modules["test_module"] = mock_module

        # 获取模块
        result = engine.get_module("test_module")
        assert result == mock_module

    async def test_get_nonexistent_module(self):
        """测试获取不存在的模块"""
        config = {"modules": {}}
        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager
        )

        result = engine.get_module("nonexistent")
        assert result is None

    async def test_conversation_manager_created(self):
        """测试 ConversationManager 被创建"""
        config = {"modules": {}}
        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager
        )

        # ConversationManager 应该在 __init__ 中被创建
        assert engine.conversation_manager is not None
        assert engine.conversation_manager.session_manager is self.session_manager

    async def test_full_initialization_flow(self):
        """测试完整初始化流程"""
        config = {
            "modules": {
                "llm": {
                    "adapter_type": "mock_llm",
                    "config": {"model": "test"}
                },
                "tts": {
                    "adapter_type": "mock_tts",
                    "config": {}
                }
            }
        }

        # 准备 Mock Loader 和 模块
        loader = AdapterLoader()
        mock_llm = MockModule("llm", {"model": "test"})
        mock_tts = MockModule("tts", {})

        # 注册 Mock 工厂
        mock_llm_factory = Mock(return_value=mock_llm)
        mock_tts_factory = Mock(return_value=mock_tts)

        loader.register("llm", mock_llm_factory)
        loader.register("tts", mock_tts_factory)

        engine = ChatEngine(
            config=config,
            session_manager=self.session_manager,
            adapter_loader=loader
        )

        await engine.initialize()

        # 验证所有模块被创建
        assert "llm" in engine.common_modules
        assert "tts" in engine.common_modules

        # 验证 AppContext 包含所有模块
        assert AppContext.get_module("llm") is not None
        assert AppContext.get_module("tts") is not None
