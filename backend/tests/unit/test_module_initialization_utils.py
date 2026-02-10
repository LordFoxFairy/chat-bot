"""module_initialization_utils 单元测试"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.utils.module_initialization_utils import (
    resolve_adapter_config,
    initialize_single_module_instance
)
from backend.core.interfaces.base_module import BaseModule
from backend.core.models.exceptions import ModuleInitializationError


class TestResolveAdapterConfig:
    """resolve_adapter_config 测试类"""

    def test_empty_config(self):
        """测试空配置"""
        result = resolve_adapter_config({})
        assert result == {}

    def test_simple_config(self):
        """测试简单配置"""
        module_config = {
            "config": {"key": "value"}
        }
        result = resolve_adapter_config(module_config)
        assert result == {"key": "value"}

    def test_enable_module_selection(self):
        """测试 enable_module 选择子配置"""
        module_config = {
            "enable_module": "openai",
            "config": {
                "openai": {"api_key": "sk-xxx"},
                "anthropic": {"api_key": "ak-xxx"}
            }
        }
        result = resolve_adapter_config(module_config)
        assert result == {"api_key": "sk-xxx"}

    def test_enable_module_not_found(self):
        """测试 enable_module 指定的配置不存在"""
        module_config = {
            "enable_module": "nonexistent",
            "config": {
                "openai": {"api_key": "sk-xxx"}
            }
        }
        result = resolve_adapter_config(module_config)
        assert result == {}

    def test_top_level_keys_merge(self):
        """测试顶层配置合并"""
        module_config = {
            "enable_module": "openai",
            "system_prompt": "You are helpful",
            "config": {
                "openai": {"api_key": "sk-xxx"}
            }
        }
        result = resolve_adapter_config(module_config)
        assert result == {"api_key": "sk-xxx", "system_prompt": "You are helpful"}

    def test_adapter_type_fallback(self):
        """测试 adapter_type 作为 key 的回退"""
        module_config = {
            "adapter_type": "openai",
            "config": {
                "openai": {"api_key": "sk-xxx"}
            }
        }
        result = resolve_adapter_config(module_config)
        assert result == {"api_key": "sk-xxx"}


class TestInitializeSingleModuleInstance:
    """initialize_single_module_instance 测试类"""

    @pytest.fixture
    def mock_module(self):
        """创建模拟模块"""
        module = MagicMock(spec=BaseModule)
        module.setup = AsyncMock()
        return module

    @pytest.fixture
    def mock_factory(self, mock_module):
        """创建模拟工厂"""
        return MagicMock(return_value=mock_module)

    @pytest.mark.asyncio
    async def test_invalid_config_type_returns_none(self):
        """测试无效配置类型返回 None"""
        result = await initialize_single_module_instance(
            module_id="test",
            module_config="invalid",  # 不是 dict
            factory_dict={},
            base_class=BaseModule,
            existing_modules={},
            raise_on_error=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_config_type_raises_when_flag_set(self):
        """测试无效配置类型时设置 raise_on_error=True 抛出异常"""
        with pytest.raises(ValueError) as exc_info:
            await initialize_single_module_instance(
                module_id="test",
                module_config="invalid",
                factory_dict={},
                base_class=BaseModule,
                existing_modules={},
                raise_on_error=True
            )
        assert "配置必须是 dict" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_existing_module_skipped(self):
        """测试已存在模块跳过初始化"""
        existing = {"test": MagicMock()}
        result = await initialize_single_module_instance(
            module_id="test",
            module_config={"adapter_type": "mock"},
            factory_dict={"test": MagicMock()},
            base_class=BaseModule,
            existing_modules=existing,
            raise_on_error=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_factory_returns_none(self):
        """测试缺少工厂返回 None"""
        result = await initialize_single_module_instance(
            module_id="test",
            module_config={"adapter_type": "mock"},
            factory_dict={},  # 空工厂字典
            base_class=BaseModule,
            existing_modules={},
            raise_on_error=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_factory_raises_when_flag_set(self):
        """测试缺少工厂时设置 raise_on_error=True 抛出异常"""
        with pytest.raises(ValueError) as exc_info:
            await initialize_single_module_instance(
                module_id="test",
                module_config={"adapter_type": "mock"},
                factory_dict={},
                base_class=BaseModule,
                existing_modules={},
                raise_on_error=True
            )
        assert "未注册模块类型" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_successful_initialization(self, mock_factory, mock_module):
        """测试成功初始化"""
        existing = {}
        result = await initialize_single_module_instance(
            module_id="test",
            module_config={"adapter_type": "mock"},
            factory_dict={"test": mock_factory},
            base_class=BaseModule,
            existing_modules=existing,
            raise_on_error=False
        )
        assert result is mock_module
        assert "test" in existing
        mock_module.setup.assert_called_once()

    @pytest.mark.asyncio
    async def test_factory_exception_returns_none(self):
        """测试工厂异常返回 None"""
        def failing_factory(**kwargs):
            raise RuntimeError("Factory failed")

        result = await initialize_single_module_instance(
            module_id="test",
            module_config={"adapter_type": "mock"},
            factory_dict={"test": failing_factory},
            base_class=BaseModule,
            existing_modules={},
            raise_on_error=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_factory_exception_raises_when_flag_set(self):
        """测试工厂异常时设置 raise_on_error=True 抛出异常"""
        def failing_factory(**kwargs):
            raise RuntimeError("Factory failed")

        with pytest.raises(RuntimeError):
            await initialize_single_module_instance(
                module_id="test",
                module_config={"adapter_type": "mock"},
                factory_dict={"test": failing_factory},
                base_class=BaseModule,
                existing_modules={},
                raise_on_error=True
            )

    @pytest.mark.asyncio
    async def test_module_initialization_error_raises_when_flag_set(self, mock_factory, mock_module):
        """测试 ModuleInitializationError 时设置 raise_on_error=True 抛出异常"""
        mock_module.setup = AsyncMock(side_effect=ModuleInitializationError("Setup failed"))

        with pytest.raises(ModuleInitializationError):
            await initialize_single_module_instance(
                module_id="test",
                module_config={"adapter_type": "mock"},
                factory_dict={"test": mock_factory},
                base_class=BaseModule,
                existing_modules={},
                raise_on_error=True
            )
