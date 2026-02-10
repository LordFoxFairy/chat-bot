"""AppContext 单元测试"""
import pytest
from backend.core.app_context import AppContext


class TestAppContext:
    """AppContext 测试类"""

    def setup_method(self):
        """每个测试前清空 AppContext"""
        AppContext.clear()

    def teardown_method(self):
        """每个测试后清空 AppContext"""
        AppContext.clear()

    def test_set_and_get_modules(self):
        """测试设置和获取模块"""
        # 模拟模块
        mock_modules = {
            "llm": "mock_llm_instance",
            "tts": "mock_tts_instance",
            "asr": "mock_asr_instance"
        }

        # 设置模块
        AppContext.set_modules(mock_modules)

        # 获取模块
        assert AppContext.get_module("llm") == "mock_llm_instance"
        assert AppContext.get_module("tts") == "mock_tts_instance"
        assert AppContext.get_module("asr") == "mock_asr_instance"

    def test_get_nonexistent_module(self):
        """测试获取不存在的模块"""
        AppContext.set_modules({"llm": "mock_llm"})

        # 获取不存在的模块应返回 None
        assert AppContext.get_module("nonexistent") is None

    def test_get_module_before_set(self):
        """测试在设置前获取模块"""
        # 未设置前应返回 None
        assert AppContext.get_module("llm") is None

    def test_clear(self):
        """测试清空上下文"""
        AppContext.set_modules({"llm": "mock_llm"})
        assert AppContext.get_module("llm") == "mock_llm"

        # 清空
        AppContext.clear()
        assert AppContext.get_module("llm") is None

    def test_override_modules(self):
        """测试覆盖模块"""
        AppContext.set_modules({"llm": "version1"})
        assert AppContext.get_module("llm") == "version1"

        # 覆盖
        AppContext.set_modules({"llm": "version2"})
        assert AppContext.get_module("llm") == "version2"
