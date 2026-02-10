import pytest
import sys
from unittest.mock import MagicMock, patch

from backend.adapters.tts.tts_factory import tts_registry, create_tts_adapter
from backend.core.interfaces.base_tts import BaseTTS
from backend.core.models.exceptions import ModuleInitializationError

# 定义一个 Mock TTS Adapter 类用于测试
class MockTTSAdapter(BaseTTS):
    async def synthesize_stream(self, text):
        yield b"audio data"

class TestTTSFactory:
    def setup_method(self):
        # 每次测试前保存原始的 loaders，以便测试后恢复
        self.original_loaders = tts_registry._loaders.copy()

    def teardown_method(self):
        # 恢复原始 loaders
        tts_registry._loaders = self.original_loaders

    def test_registry_initialization(self):
        """测试 TTS 注册表初始化"""
        assert tts_registry.name == "TTS"
        assert tts_registry._base_class == BaseTTS
        # 验证默认已注册 edge_tts
        assert "edge_tts" in tts_registry.available_types

    def test_register_class(self):
        """测试注册新的适配器类"""
        tts_registry.register_class("mock_tts", MockTTSAdapter)
        assert "mock_tts" in tts_registry.available_types

        # 测试创建实例
        adapter = create_tts_adapter("mock_tts", "test_id", {})
        assert isinstance(adapter, MockTTSAdapter)
        assert adapter.module_id == "test_id"

    def test_create_unknown_adapter(self):
        """测试创建未知的适配器"""
        with pytest.raises(ModuleInitializationError, match="不支持的 TTS 适配器类型"):
            create_tts_adapter("unknown_type", "test_id", {})

    def test_register_function_load(self):
        """测试通过模块路径注册（使用 load 函数）"""
        # 模拟 import_module
        mock_module = MagicMock()
        mock_module.load.return_value = MockTTSAdapter

        with patch("src.core.adapter_registry.import_module", return_value=mock_module):
            tts_registry.register("test_module_load", "some.module.path")

            adapter = create_tts_adapter("test_module_load", "test_id", {})
            assert isinstance(adapter, MockTTSAdapter)
            mock_module.load.assert_called_once()

    def test_register_class_path(self):
        """测试通过模块路径注册（直接指定类名）"""
        # 模拟 import_module
        mock_module = MagicMock()
        mock_module.MockTTSAdapter = MockTTSAdapter

        with patch("src.core.adapter_registry.import_module", return_value=mock_module):
            tts_registry.register("test_class_path", "some.module.path:MockTTSAdapter")

            adapter = create_tts_adapter("test_class_path", "test_id", {})
            assert isinstance(adapter, MockTTSAdapter)
