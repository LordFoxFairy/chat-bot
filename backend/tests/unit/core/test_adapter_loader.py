import pytest
from typing import Dict, Any, Type
from unittest.mock import Mock, patch

from backend.core.adapter_loader import AdapterLoader, create_default_loader
from backend.core.interfaces.base_module import BaseModule


class MockAdapter(BaseModule):
    """用于测试的 Mock 适配器"""

    async def _setup_impl(self) -> None:
        pass

    async def _close_impl(self) -> None:
        pass


def create_mock_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
    **kwargs: Any
) -> BaseModule:
    """创建 Mock 适配器的工厂函数"""
    return MockAdapter(module_id=module_id, config=config)


class TestAdapterLoader:

    def test_register_and_create(self):
        """测试注册和创建适配器"""
        loader = AdapterLoader()
        module_type = "test_module"

        # 1. 测试注册
        loader.register(module_type, create_mock_adapter)

        assert loader.has_factory(module_type)
        assert module_type in loader.registered_types

        # 2. 测试创建
        adapter = loader.create(
            module_type=module_type,
            adapter_type="test_adapter",
            module_id="test_id",
            config={"key": "value"}
        )

        assert isinstance(adapter, MockAdapter)
        assert adapter.module_id == "test_id"
        assert adapter.config == {"key": "value"}

    def test_create_unregistered_module(self):
        """测试创建未注册模块，应抛出异常"""
        loader = AdapterLoader()

        with pytest.raises(ValueError) as excinfo:
            loader.create(
                module_type="unknown_module",
                adapter_type="default",
                module_id="test",
                config={}
            )

        assert "未注册的模块类型: unknown_module" in str(excinfo.value)

    def test_factory_parameters_passing(self):
        """测试工厂函数参数透传"""
        loader = AdapterLoader()

        # 捕获参数的 Mock 工厂
        mock_factory = Mock(return_value=MockAdapter("id", {}))
        loader.register("test", mock_factory)

        extra_kwargs = {"extra_param": 123}
        loader.create(
            module_type="test",
            adapter_type="my_adapter",
            module_id="my_id",
            config={"conf": "val"},
            **extra_kwargs
        )

        # 验证参数是否正确传递给工厂
        mock_factory.assert_called_once_with(
            adapter_type="my_adapter",
            module_id="my_id",
            config={"conf": "val"},
            extra_param=123
        )

    def test_chain_registration(self):
        """测试链式注册"""
        loader = AdapterLoader()

        (loader.register("type1", create_mock_adapter)
               .register("type2", create_mock_adapter))

        assert loader.has_factory("type1")
        assert loader.has_factory("type2")
        assert len(loader.registered_types) == 2


class TestCreateDefaultLoader:

    def test_create_default_loader(self):
        """测试默认加载器包含预期模块"""

        # Mock 所有工厂导入，避免依赖实际实现
        with patch.dict('sys.modules', {
            'backend.adapters.asr.asr_factory': Mock(create_asr_adapter=Mock()),
            'backend.adapters.llm.llm_factory': Mock(create_llm_adapter=Mock()),
            'backend.adapters.protocols.protocol_factory': Mock(create_protocol_adapter=Mock()),
            'backend.adapters.tts.tts_factory': Mock(create_tts_adapter=Mock()),
            'backend.adapters.vad.vad_factory': Mock(create_vad_adapter=Mock()),
        }):
            loader = create_default_loader()

            expected_modules = ["asr", "llm", "tts", "vad", "protocol"]

            for module in expected_modules:
                assert loader.has_factory(module), f"默认加载器缺少 {module} 工厂"

            assert len(loader.registered_types) == len(expected_modules)
