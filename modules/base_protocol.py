from abc import abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import uuid

from modules.base_module import BaseModule
from utils.logging_setup import logger


# 泛型：连接类型
ConnectionT = TypeVar('ConnectionT')


class BaseProtocol(BaseModule, Generic[ConnectionT]):
    """通信协议模块基类

    职责:
    - 定义协议核心接口
    - 提供通用的会话管理
    - 提供通用的连接映射

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

        # 通用会话映射
        self.tag_to_session: Dict[str, str] = {}
        self.session_to_connection: Dict[str, ConnectionT] = {}
        self.connection_to_session: Dict[ConnectionT, str] = {}

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

    # ==================== 通用会话管理 ====================

    def create_session(
        self,
        connection: ConnectionT,
        tag_id: Optional[str] = None
    ) -> str:
        """创建新会话并建立映射"""
        session_id = str(uuid.uuid4())

        # 建立映射
        if tag_id:
            self.tag_to_session[tag_id] = session_id

        self.session_to_connection[session_id] = connection
        self.connection_to_session[connection] = session_id

        logger.debug(
            f"Protocol [{self.module_id}] 创建会话: "
            f"session={session_id}, tag={tag_id}"
        )

        return session_id

    def get_session_id(self, connection: ConnectionT) -> Optional[str]:
        """通过连接获取会话 ID"""
        return self.connection_to_session.get(connection)

    def get_connection(self, session_id: str) -> Optional[ConnectionT]:
        """通过会话 ID 获取连接"""
        return self.session_to_connection.get(session_id)

    def remove_session(self, session_id: str):
        """移除会话映射"""
        connection = self.session_to_connection.pop(session_id, None)
        if connection:
            self.connection_to_session.pop(connection, None)

        # 移除 tag 映射
        tag_to_remove = None
        for tag, sid in self.tag_to_session.items():
            if sid == session_id:
                tag_to_remove = tag
                break
        if tag_to_remove:
            self.tag_to_session.pop(tag_to_remove, None)

        logger.debug(
            f"Protocol [{self.module_id}] 移除会话: {session_id}"
        )

    def remove_session_by_connection(self, connection: ConnectionT):
        """通过连接移除会话"""
        session_id = self.connection_to_session.pop(connection, None)
        if session_id:
            self.session_to_connection.pop(session_id, None)

            # 移除 tag 映射
            tag_to_remove = None
            for tag, sid in self.tag_to_session.items():
                if sid == session_id:
                    tag_to_remove = tag
                    break
            if tag_to_remove:
                self.tag_to_session.pop(tag_to_remove, None)

            logger.debug(
                f"Protocol [{self.module_id}] 移除会话: {session_id}"
            )

        return session_id

    def clear_all_sessions(self):
        """清理所有会话映射"""
        logger.debug(f"Protocol [{self.module_id}] 清理所有会话映射")
        self.tag_to_session.clear()
        self.session_to_connection.clear()
        self.connection_to_session.clear()
