"""VAD Factory 测试"""
import pytest
from unittest.mock import MagicMock, patch

from backend.adapters.vad.vad_factory import vad_registry, create_vad_adapter
from backend.core.interfaces.base_vad import BaseVAD


class TestVADFactory:

    def test_registry_configuration(self):
        """测试注册表配置正确"""
        # 检查注册器名称
        assert vad_registry.name == "VAD"

        # 验证 silero_vad 已注册
        assert vad_registry.is_registered("silero_vad")
        assert "silero_vad" in vad_registry.available_types

    def test_create_vad_adapter(self):
        """测试工厂函数调用"""
        mock_instance = MagicMock()

        with patch.object(vad_registry, 'create', return_value=mock_instance) as mock_create:
            config = {"some": "config"}
            result = create_vad_adapter("silero_vad", "vad_id", config)

            assert result == mock_instance
            mock_create.assert_called_once_with("silero_vad", "vad_id", config)

    def test_create_factory_function_export(self):
        """验证 create_vad_adapter 被正确导出"""
        # 确保它是一个可调用对象
        assert callable(create_vad_adapter)
