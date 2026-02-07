"""适配器加载器

提供统一的适配器工厂注册和创建接口，解耦 ChatEngine 与 adapters 层。

使用方式:
    # 创建加载器
    loader = AdapterLoader()

    # 注册工厂（在应用启动时）
    loader.register("asr", create_asr_adapter)
    loader.register("llm", create_llm_adapter)

    # 创建适配器
    asr = loader.create("asr", adapter_type, module_id, config)
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

from src.core.interfaces.base_module import BaseModule
from src.utils.logging_setup import logger

T = TypeVar("T", bound=BaseModule)

# 工厂函数类型
FactoryFunction = Callable[..., BaseModule]


class AdapterLoader:
    """适配器加载器

    统一管理适配器工厂，解耦核心层与适配器层。
    """

    def __init__(self) -> None:
        self._factories: Dict[str, FactoryFunction] = {}

    def register(
        self,
        module_type: str,
        factory: FactoryFunction,
    ) -> "AdapterLoader":
        """注册适配器工厂

        Args:
            module_type: 模块类型 (asr, llm, tts, vad, protocol)
            factory: 工厂函数

        Returns:
            self，支持链式调用
        """
        self._factories[module_type] = factory
        logger.debug(f"AdapterLoader: 注册 {module_type} 工厂")
        return self

    def create(
        self,
        module_type: str,
        adapter_type: str,
        module_id: str,
        config: Dict[str, Any],
        **kwargs: Any,
    ) -> BaseModule:
        """创建适配器实例

        Args:
            module_type: 模块类型
            adapter_type: 适配器类型
            module_id: 模块 ID
            config: 配置
            **kwargs: 额外参数

        Returns:
            适配器实例

        Raises:
            ValueError: 未注册的模块类型
        """
        factory = self._factories.get(module_type)
        if not factory:
            available = list(self._factories.keys())
            raise ValueError(
                f"未注册的模块类型: {module_type}，可用类型: {available}"
            )

        return factory(
            adapter_type=adapter_type,
            module_id=module_id,
            config=config,
            **kwargs,
        )

    def has_factory(self, module_type: str) -> bool:
        """检查是否已注册工厂"""
        return module_type in self._factories

    @property
    def registered_types(self) -> list[str]:
        """获取所有已注册的模块类型"""
        return list(self._factories.keys())


def create_default_loader() -> AdapterLoader:
    """创建默认的适配器加载器（注册所有内置适配器）

    Returns:
        配置好的 AdapterLoader 实例
    """
    from src.adapters.asr.asr_factory import create_asr_adapter
    from src.adapters.llm.llm_factory import create_llm_adapter
    from src.adapters.protocols.protocol_factory import create_protocol_adapter
    from src.adapters.tts.tts_factory import create_tts_adapter
    from src.adapters.vad.vad_factory import create_vad_adapter

    loader = AdapterLoader()
    loader.register("asr", create_asr_adapter)
    loader.register("llm", create_llm_adapter)
    loader.register("tts", create_tts_adapter)
    loader.register("vad", create_vad_adapter)
    loader.register("protocol", create_protocol_adapter)

    return loader
