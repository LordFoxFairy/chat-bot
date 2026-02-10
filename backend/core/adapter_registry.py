"""通用适配器注册器模块

提供统一的工厂模式实现，消除各模块工厂的代码重复。

使用方式:
    # 创建注册器
    asr_registry = AdapterRegistry("ASR", BaseASR)

    # 注册适配器
    asr_registry.register("funasr_sensevoice", "adapters.asr.funasr_sensevoice_adapter:FunASRSenseVoiceAdapter")

    # 创建实例
    instance = asr_registry.create("funasr_sensevoice", module_id="asr", config={})
"""

from importlib import import_module
from typing import Any, Callable, Dict, Generic, Optional, Type, TypeVar

from backend.core.models.exceptions import ModuleInitializationError
from backend.utils.logging_setup import logger

# 泛型类型变量，用于类型提示
T = TypeVar("T")


class AdapterRegistry(Generic[T]):
    """通用适配器注册器

    支持延迟加载、类型检查和统一的错误处理。

    Attributes:
        name: 注册器名称，用于日志和错误消息
        base_class: 适配器基类，用于类型检查
    """

    def __init__(self, name: str, base_class: Type[T]):
        """初始化注册器

        Args:
            name: 注册器名称 (如 "ASR", "LLM", "TTS")
            base_class: 适配器基类 (如 BaseASR, BaseLLM)
        """
        self._name = name
        self._base_class = base_class
        self._loaders: Dict[str, Callable[[], Type[T]]] = {}

    @property
    def name(self) -> str:
        """注册器名称"""
        return self._name

    @property
    def available_types(self) -> list[str]:
        """获取所有已注册的适配器类型"""
        return list(self._loaders.keys())

    def register(self, adapter_type: str, loader_path: str) -> "AdapterRegistry[T]":
        """注册适配器

        Args:
            adapter_type: 适配器类型标识符 (如 "funasr_sensevoice")
            loader_path: 模块路径和类名，格式为 "module.path:ClassName"
                        或 "module.path" (将调用模块的 load() 函数)

        Returns:
            self，支持链式调用

        Example:
            registry.register("edge_tts", "adapters.tts.edge_tts_adapter:EdgeTTSAdapter")
            registry.register("funasr", "adapters.asr.funasr_adapter")  # 使用 load() 函数
        """
        def _loader() -> Type[T]:
            if ":" in loader_path:
                # 直接指定类名: "module.path:ClassName"
                module_path, class_name = loader_path.rsplit(":", 1)
                module = import_module(module_path)
                return getattr(module, class_name)
            else:
                # 使用 load() 函数: "module.path"
                module = import_module(loader_path)
                return module.load()

        self._loaders[adapter_type] = _loader
        return self

    def register_class(self, adapter_type: str, adapter_class: Type[T]) -> "AdapterRegistry[T]":
        """直接注册适配器类（非延迟加载）

        Args:
            adapter_type: 适配器类型标识符
            adapter_class: 适配器类

        Returns:
            self，支持链式调用
        """
        self._loaders[adapter_type] = lambda: adapter_class
        return self

    def create(
        self,
        adapter_type: str,
        module_id: str,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> T:
        """创建适配器实例

        Args:
            adapter_type: 适配器类型标识符
            module_id: 模块唯一标识符
            config: 模块配置
            **kwargs: 传递给适配器构造函数的额外参数

        Returns:
            适配器实例

        Raises:
            ModuleInitializationError: 当适配器类型不支持或创建失败时
        """
        loader = self._loaders.get(adapter_type)

        if loader is None:
            raise ModuleInitializationError(
                f"不支持的 {self._name} 适配器类型: '{adapter_type}'. "
                f"可用类型: {self.available_types}"
            )

        try:
            adapter_class = loader()

            # 类型检查
            if not issubclass(adapter_class, self._base_class):
                raise ModuleInitializationError(
                    f"适配器 '{adapter_type}' 的类 {adapter_class.__name__} "
                    f"不是 {self._base_class.__name__} 的子类"
                )

            logger.info(
                f"{self._name} Factory: 创建 '{adapter_type}' 适配器，"
                f"模块ID: {module_id}，类: {adapter_class.__name__}"
            )

            return adapter_class(
                module_id=module_id,
                config=config,
                **kwargs
            )

        except ModuleInitializationError:
            raise
        except ImportError as e:
            raise ModuleInitializationError(
                f"导入 {self._name} 适配器 '{adapter_type}' 失败: {e}"
            ) from e
        except Exception as e:
            raise ModuleInitializationError(
                f"创建 {self._name} 适配器 '{adapter_type}' 失败: {e}"
            ) from e

    def is_registered(self, adapter_type: str) -> bool:
        """检查适配器类型是否已注册"""
        return adapter_type in self._loaders

    def unregister(self, adapter_type: str) -> bool:
        """注销适配器类型

        Returns:
            是否成功注销
        """
        if adapter_type in self._loaders:
            del self._loaders[adapter_type]
            return True
        return False


def create_factory_function(
    registry: AdapterRegistry[T],
    factory_name: Optional[str] = None
) -> Callable[..., T]:
    """创建工厂函数（向后兼容）

    为已有代码提供与旧 API 兼容的工厂函数。

    Args:
        registry: 适配器注册器
        factory_name: 工厂函数名称，用于日志

    Returns:
        工厂函数

    Example:
        create_asr_adapter = create_factory_function(asr_registry, "ASR")
    """
    def factory(
        adapter_type: str,
        module_id: str,
        config: Dict[str, Any],
        **kwargs: Any
    ) -> T:
        return registry.create(adapter_type, module_id, config, **kwargs)

    factory.__name__ = f"create_{registry.name.lower()}_adapter"
    factory.__doc__ = f"创建 {registry.name} 适配器实例"

    return factory
