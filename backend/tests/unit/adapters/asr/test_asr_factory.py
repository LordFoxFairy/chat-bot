import pytest
import sys
from unittest.mock import MagicMock, patch
from backend.adapters.asr.asr_factory import asr_registry, create_asr_adapter
from backend.core.interfaces.base_asr import BaseASR
from backend.core.models.exceptions import ModuleInitializationError

# 创建一个真正的类用于测试 issubclass
class MockAdapterClass(BaseASR):
    def __init__(self, module_id, config):
        pass

    async def recognize(self, audio):
        pass

def test_asr_registry_config():
    """测试 ASR 注册器配置"""
    assert asr_registry.name == "ASR"
    assert asr_registry._base_class == BaseASR
    assert "funasr_sensevoice" in asr_registry.available_types

def test_create_asr_adapter_factory():
    """测试工厂函数"""
    # 模拟 import_module 以避免实际加载适配器
    with patch('src.core.adapter_registry.import_module') as mock_import:
        # 准备 mock 模块
        mock_module = MagicMock()

        # 使用真实的类而不是 MagicMock，因为 issubclass 需要一个类
        mock_module.load.return_value = MockAdapterClass
        mock_import.return_value = mock_module

        # 调用工厂函数
        adapter = create_asr_adapter(
            "funasr_sensevoice",
            "test_asr",
            {"model_dir": "test"}
        )

        # 验证
        assert isinstance(adapter, MockAdapterClass)
        mock_import.assert_called_with("src.adapters.asr.funasr_sensevoice_adapter")
        mock_module.load.assert_called_once()


def test_create_unknown_adapter():
    """测试创建未知适配器"""
    with pytest.raises(ModuleInitializationError, match="不支持的 ASR 适配器类型"):
        create_asr_adapter("unknown_type", "test_id", {})
