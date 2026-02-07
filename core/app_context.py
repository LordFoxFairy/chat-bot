"""应用全局上下文

职责:
- 提供全局模块访问
- 存储应用级共享数据
- 与具体业务逻辑解耦

使用方式:
    # 设置模块（应用启动时）
    AppContext.set_modules({"llm": llm_instance, "tts": tts_instance})

    # 获取模块
    llm = AppContext.get_module("llm")

    # 获取模块（带类型检查）
    from modules.base_llm import BaseLLM
    llm = AppContext.get_module_typed("llm", BaseLLM)
"""

from threading import RLock
from typing import Any, Dict, Optional, Type, TypeVar, overload

T = TypeVar("T")


class AppContext:
    """应用全局上下文

    提供全局模块访问，避免层层传递依赖。
    支持线程安全和类型检查。
    """

    # 全局模块字典
    _modules: Dict[str, Any] = {}
    _lock: RLock = RLock()

    @classmethod
    def set_modules(cls, modules: Dict[str, Any]) -> None:
        """设置全局模块（仅在应用启动时调用一次）

        Args:
            modules: 模块字典 {name: module_instance}
        """
        with cls._lock:
            cls._modules = modules.copy()

    @classmethod
    def set_module(cls, name: str, module: Any) -> None:
        """设置单个模块

        Args:
            name: 模块名称
            module: 模块实例
        """
        with cls._lock:
            cls._modules[name] = module

    @classmethod
    def get_module(cls, name: str) -> Optional[Any]:
        """获取模块实例

        Args:
            name: 模块名称 (asr, llm, tts, vad)

        Returns:
            模块实例，找不到返回 None
        """
        with cls._lock:
            return cls._modules.get(name)

    @classmethod
    def get_module_typed(cls, name: str, module_type: Type[T]) -> Optional[T]:
        """获取模块实例（带类型检查）

        Args:
            name: 模块名称
            module_type: 期望的模块类型

        Returns:
            模块实例，找不到或类型不匹配返回 None

        Example:
            llm = AppContext.get_module_typed("llm", BaseLLM)
        """
        with cls._lock:
            module = cls._modules.get(name)
            if module is not None and isinstance(module, module_type):
                return module
            return None

    @classmethod
    def has_module(cls, name: str) -> bool:
        """检查模块是否存在

        Args:
            name: 模块名称

        Returns:
            模块是否存在
        """
        with cls._lock:
            return name in cls._modules

    @classmethod
    def get_all_modules(cls) -> Dict[str, Any]:
        """获取所有模块的副本

        Returns:
            模块字典副本
        """
        with cls._lock:
            return cls._modules.copy()

    @classmethod
    def remove_module(cls, name: str) -> Optional[Any]:
        """移除模块

        Args:
            name: 模块名称

        Returns:
            被移除的模块实例，不存在返回 None
        """
        with cls._lock:
            return cls._modules.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        """清空全局上下文（测试用）"""
        with cls._lock:
            cls._modules.clear()
