from abc import ABC, abstractmethod
from typing import Dict, Any

from modules.base_module import BaseModule
from utils.logging_setup import logger


class BaseProtocol(BaseModule):
    """通信协议模块基类

    职责:
    - 定义协议核心接口
    - 提供通用的连接管理

    子类需要实现:
    - setup: 初始化协议服务
    - start: 启动协议服务
    - stop: 停止协议服务
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取协议通用配置
        self.host = self.config.get("host", "0.0.0.0")
        self.port = self.config.get("port", 8765)

        logger.debug(f"Protocol [{self.module_id}] 配置加载:")
        logger.debug(f"  - host: {self.host}")
        logger.debug(f"  - port: {self.port}")

    @abstractmethod
    async def start(self):
        """启动协议服务"""
        raise NotImplementedError("Protocol 子类必须实现 start 方法")

    @abstractmethod
    async def stop(self):
        """停止协议服务"""
        raise NotImplementedError("Protocol 子类必须实现 stop 方法")
