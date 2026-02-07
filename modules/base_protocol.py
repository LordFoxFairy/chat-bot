from abc import abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
import uuid

from modules.base_module import BaseModule
from models import StreamEvent, EventType, TextData
from utils.logging_setup import logger


# 泛型：连接类型
ConnectionT = TypeVar('ConnectionT')


class BaseProtocol(BaseModule, Generic[ConnectionT]):
    """通信协议模块基类

    职责:
    - 定义协议核心接口
    - 提供通用的会话管理
    - 提供通用的协议消息处理（注册、路由、断开）
    - 子类只需实现具体传输层

    子类需要实现:
    - setup: 初始化协议服务
    - start: 启动协议服务
    - stop: 停止协议服务
    - send_message: 发送消息到连接
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
        conversation_manager: 'ConversationManager'
    ):
        super().__init__(module_id, config)

        # ConversationManager 引用
        self.conversation_manager = conversation_manager

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

    async def _setup_impl(self):
        """初始化逻辑（默认为空，子类可覆盖）"""
        pass

    # ==================== 通用协议消息处理 ====================

    async def handle_text_message(self, connection: ConnectionT, raw_message: str):
        """处理文本消息（通用方法）"""
        try:
            if not raw_message.strip().startswith('{'):
                return

            stream_event = StreamEvent.model_validate_json(raw_message)

            # 判断是否为注册消息
            if stream_event.event_type == EventType.SYSTEM_CLIENT_SESSION_START:
                await self._handle_register(connection, stream_event)
            else:
                # 路由到 ConversationHandler
                await self._route_message(connection, stream_event)

        except Exception as e:
            logger.error(f"Protocol [{self.module_id}] 消息处理失败: {e}")

    async def _handle_register(self, connection: ConnectionT, stream_event: StreamEvent):
        """处理注册消息（通用方法）"""
        from core.session_context import SessionContext

        tag_id = stream_event.tag_id

        # 创建会话映射
        session_id = self.create_session(connection, tag_id)

        # 创建 SessionContext（模块通过 AppContext 全局访问）
        session_ctx = SessionContext(
            session_id=session_id,
            tag_id=tag_id
        )

        # 调用 ConversationManager 创建 ConversationHandler
        await self.conversation_manager.create_conversation_handler(
            session_id=session_id,
            tag_id=tag_id,
            send_callback=lambda event: self.send_event(session_id, event),
            session_context=session_ctx
        )

        # 发送会话确认
        response = StreamEvent(
            event_type=EventType.SYSTEM_SERVER_SESSION_START,
            tag_id=tag_id,
            session_id=session_id
        )
        await self.send_event(session_id, response)

        logger.info(f"Protocol [{self.module_id}] 客户端已注册: tag={tag_id}, session={session_id}")

    async def _route_message(self, connection: ConnectionT, stream_event: StreamEvent):
        """路由消息到 ConversationHandler（通用方法）"""
        session_id = self.get_session_id(connection)
        if not session_id:
            logger.warning(f"Protocol [{self.module_id}] 未找到会话映射")
            return

        handler = self.conversation_manager.get_conversation_handler(session_id)
        if not handler:
            logger.warning(f"Protocol [{self.module_id}] 会话处理器不存在: {session_id}")
            return

        # 根据事件类型分发
        if stream_event.event_type == EventType.CLIENT_SPEECH_END:
            await handler.handle_speech_end()
        elif stream_event.event_type == EventType.STREAM_END:
            await handler.handle_speech_end()
        elif stream_event.event_type == EventType.CLIENT_TEXT_INPUT:
            text_data: TextData = stream_event.event_data
            await handler.handle_text_input(text_data.text)

    async def handle_audio_message(self, connection: ConnectionT, audio_data: bytes):
        """处理音频消息（通用方法）"""
        session_id = self.get_session_id(connection)
        if not session_id:
            return

        handler = self.conversation_manager.get_conversation_handler(session_id)
        if handler:
            await handler.handle_audio(audio_data)

    async def handle_disconnect(self, connection: ConnectionT):
        """处理断开连接（通用方法）"""
        session_id = self.remove_session_by_connection(connection)
        if session_id:
            logger.info(f"Protocol [{self.module_id}] 连接断开: session={session_id}")
            await self.conversation_manager.destroy_conversation_handler(session_id)

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

    # ==================== 消息发送 ====================

    @abstractmethod
    async def send_message(self, connection: ConnectionT, message: str):
        """发送消息到连接（子类实现具体传输逻辑）"""
        raise NotImplementedError("Protocol 子类必须实现 send_message 方法")

    async def send_event(
        self,
        session_id: str,
        event: StreamEvent
    ) -> bool:
        """发送事件到会话（通用方法）"""
        connection = self.get_connection(session_id)
        if not connection:
            logger.warning(
                f"Protocol [{self.module_id}] 会话 {session_id} 的连接不存在"
            )
            return False

        try:
            await self.send_message(connection, event.to_json())
            return True
        except Exception as e:
            logger.error(
                f"Protocol [{self.module_id}] 发送事件失败 "
                f"(session: {session_id}): {e}"
            )
            return False
