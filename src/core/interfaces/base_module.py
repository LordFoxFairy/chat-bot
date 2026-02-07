"""模块基类定义"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModule(ABC):
    """所有模块的基类

    提供模块生命周期管理和基础功能。

    职责:
    - 生命周期管理 (setup/close)
    - 状态管理 (is_ready)
    - 配置访问辅助

    生命周期:
        1. __init__: 读取配置，初始化变量
        2. setup(): 异步初始化资源（调用 _setup_impl）
        3. is_ready = True: 可以处理请求
        4. close(): 释放资源（调用 _close_impl）
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        """初始化基础模块"""
        self.module_id = module_id
        self.config = config
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """模块是否准备好处理请求"""
        return self._is_ready

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键名
            default: 默认值

        Returns:
            配置值或默认值
        """
        return self.config.get(key, default)

    def require_config(self, key: str) -> Any:
        """获取必需的配置值

        Args:
            key: 配置键名

        Returns:
            配置值

        Raises:
            ValueError: 配置项不存在
        """
        value = self.config.get(key)
        if value is None:
            raise ValueError(f"缺少必需的配置项: {key}")
        return value

    async def setup(self) -> None:
        """初始化模块资源（模板方法）"""
        if self._is_ready:
            return

        await self._setup_impl()
        self._is_ready = True

    @abstractmethod
    async def _setup_impl(self) -> None:
        """具体初始化逻辑（由子类实现）"""
        pass

    async def close(self) -> None:
        """关闭模块，释放资源（模板方法）"""
        if not self._is_ready:
            return

        await self._close_impl()
        self._is_ready = False

    async def _close_impl(self) -> None:
        """具体关闭逻辑（由子类覆盖）"""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """健康检查

        Returns:
            健康状态信息
        """
        return {
            "module_id": self.module_id,
            "is_ready": self.is_ready,
            "status": "healthy" if self.is_ready else "not_ready",
        }

    async def __aenter__(self) -> "BaseModule":
        """异步上下文管理器入口"""
        await self.setup()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器出口"""
        await self.close()
