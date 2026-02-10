import pytest
import sys
from typing import Dict, Any, Type
from unittest.mock import Mock, patch

from backend.core.adapter_registry import AdapterRegistry
from backend.core.models.exceptions import ModuleInitializationError


class MockAdapterBase:
    """Mock 适配器基类"""
    def __init__(self, module_id: str, config: dict):
        self.module_id = module_id
        self.config = config


class MockAdapterImpl(MockAdapterBase):
    """Mock 适配器实现"""
    pass


class InvalidAdapter:
    """无效的适配器实现（未继承基类）"""
    pass


class TestAdapterRegistry:

    def setup_method(self):
        self.registry = AdapterRegistry("TEST", MockAdapterBase)

    def test_register_and_create_instance(self):
        """测试注册和创建实例（正常流程）"""
        self.registry.register_class("mock", MockAdapterImpl)

        instance = self.registry.create(
            adapter_type="mock",
            module_id="test_id",
            config={"key": "value"}
        )

        assert isinstance(instance, MockAdapterImpl)
        assert instance.module_id == "test_id"
        assert instance.config == {"key": "value"}

    def test_create_unknown_type(self):
        """测试未知适配器类型"""
        with pytest.raises(ModuleInitializationError) as excinfo:
            self.registry.create("unknown", "id", {})

        assert "不支持的 TEST 适配器类型: 'unknown'" in str(excinfo.value)

    def test_create_invalid_subclass(self):
        """测试适配器未继承基类"""
        self.registry.register_class("invalid", InvalidAdapter)

        with pytest.raises(ModuleInitializationError) as excinfo:
            self.registry.create("invalid", "id", {})

        assert "不是 MockAdapterBase 的子类" in str(excinfo.value)

    def test_dynamic_import_class(self):
        """测试动态导入类"""
        with patch('src.core.adapter_registry.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MyAdapter = MockAdapterImpl
            mock_import.return_value = mock_module

            self.registry.register("dynamic", "my.module:MyAdapter")
            instance = self.registry.create("dynamic", "id", {})

            assert isinstance(instance, MockAdapterImpl)
            mock_import.assert_called_with("my.module")

    def test_dynamic_import_loader(self):
        """测试动态导入加载函数"""
        with patch('src.core.adapter_registry.import_module') as mock_import:
            mock_module = Mock()
            mock_module.load = Mock(return_value=MockAdapterImpl)
            mock_import.return_value = mock_module

            # 无冒号格式，使用 load() 函数
            self.registry.register("loader", "my.module")
            instance = self.registry.create("loader", "id", {})

            assert isinstance(instance, MockAdapterImpl)
            mock_module.load.assert_called_once()

    def test_import_error(self):
        """测试导入失败"""
        with patch('src.core.adapter_registry.import_module', side_effect=ImportError("No module")):
            self.registry.register("missing", "missing.module:Class")

            with pytest.raises(ModuleInitializationError) as excinfo:
                self.registry.create("missing", "id", {})

            assert "导入 TEST 适配器 'missing' 失败" in str(excinfo.value)

    def test_available_types(self):
        """测试获取可用类型"""
        self.registry.register_class("a1", MockAdapterImpl)
        self.registry.register_class("a2", MockAdapterImpl)

        types = self.registry.available_types
        assert "a1" in types
        assert "a2" in types
        assert len(types) == 2

    def test_unregister(self):
        """测试注销"""
        self.registry.register_class("temp", MockAdapterImpl)
        assert self.registry.is_registered("temp")

        result = self.registry.unregister("temp")
        assert result is True
        assert not self.registry.is_registered("temp")

        # 注销不存在的
        result = self.registry.unregister("temp")
        assert result is False

