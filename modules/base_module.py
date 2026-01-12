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
    ) -> None:
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

    @abstractmethod
    async def setup(self) -> None:
        """初始化模块资源，子类必须实现此方法"""
        raise NotImplementedError("子类必须实现 setup 方法")

    async def close(self) -> None:
        """关闭模块，释放资源"""
        self._is_ready = False
