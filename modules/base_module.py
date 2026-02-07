from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModule(ABC):
    """所有模块的基类

    提供模块生命周期管理和基础功能。

    职责:
    - 生命周期管理 (setup/close)
    - 状态管理 (is_initialized/is_ready)
    - 配置访问辅助
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        """初始化基础模块"""
        self.module_id = module_id
        self.config = config

        # 初始状态为未就绪，需要调用 setup() 后才就绪
        self._is_initialized = False
        self._is_ready = False

    @property
    def is_initialized(self) -> bool:
        """模块是否已完成初始化"""
        return self._is_initialized

    @property
    def is_ready(self) -> bool:
        """模块是否准备好处理请求"""
        return self._is_ready

    async def setup(self):
        """初始化模块资源（模板方法）"""
        if self._is_initialized:
            return

        await self._setup_impl()
        self._is_initialized = True
        self._is_ready = True

    @abstractmethod
    async def _setup_impl(self):
        """具体初始化逻辑（由子类实现）"""
        pass

    async def close(self):
        """关闭模块，释放资源"""
        self._is_ready = False
        self._is_initialized = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
