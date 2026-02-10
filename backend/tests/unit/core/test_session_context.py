"""SessionContext 单元测试"""
import pytest
from backend.core.session.session_context import SessionContext
from backend.core.app_context import AppContext


class TestSessionContext:
    """SessionContext 测试类"""

    def setup_method(self):
        """每个测试前清空 AppContext"""
        AppContext.clear()

    def teardown_method(self):
        """每个测试后清空 AppContext"""
        AppContext.clear()

    def test_create_session_context(self):
        """测试创建会话上下文"""
        ctx = SessionContext(
            session_id="test_session_123",
            tag_id="user_456"
        )

        assert ctx.session_id == "test_session_123"
        assert ctx.tag_id == "user_456"
        assert ctx.dialogues == []
        assert ctx.config == {}
        assert ctx.custom_modules == {}

    def test_get_module_from_module_provider(self):
        """测试从模块提供者获取模块"""
        # 设置全局模块
        AppContext.set_modules({
            "llm": "global_llm",
            "tts": "global_tts"
        })

        ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )
        # 注入模块提供者
        ctx.set_module_provider(AppContext.get_module)

        # 应该从模块提供者获取
        assert ctx.get_module("llm") == "global_llm"
        assert ctx.get_module("tts") == "global_tts"

    def test_custom_module_override(self):
        """测试自定义模块覆盖模块提供者"""
        # 设置全局模块
        AppContext.set_modules({"llm": "global_llm"})

        # 创建会话，使用自定义模块
        ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag",
            custom_modules={"llm": "custom_llm"}
        )
        # 注入模块提供者
        ctx.set_module_provider(AppContext.get_module)

        # 应该返回自定义模块（优先于提供者）
        assert ctx.get_module("llm") == "custom_llm"

    def test_get_nonexistent_module(self):
        """测试获取不存在的模块"""
        ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag"
        )

        # 不存在的模块应返回 None
        assert ctx.get_module("nonexistent") is None

    def test_dialogues_and_config(self):
        """测试对话历史和配置"""
        ctx = SessionContext(
            session_id="test_session",
            tag_id="test_tag",
            dialogues=["hello", "world"],
            config={"key": "value"}
        )

        assert ctx.dialogues == ["hello", "world"]
        assert ctx.config == {"key": "value"}
