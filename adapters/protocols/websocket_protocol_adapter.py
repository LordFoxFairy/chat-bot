from typing import Optional, Dict, Tuple

import websockets
from websockets.server import WebSocketServerProtocol

from data_models import StreamEvent, EventType
from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger


class WebSocketProtocolAdapter(BaseProtocol[WebSocketServerProtocol]):
    """WebSocket 协议适配器

    职责:
    - 管理 WebSocket 连接
    - 接收/发送消息
    - 调用 ChatEngine 创建/销毁 ConversationHandler
    - 纯传输层，不包含业务逻辑
    """

    def __init__(self, module_id: str, config: Dict, chat_engine: 'ChatEngine'):
        super().__init__(module_id, config)

        self.chat_engine = chat_engine
        self.server: Optional[websockets.WebSocketServer] = None

        logger.info(
            f"Protocol/WebSocket [{self.module_id}] 配置加载完成: "
            f"{self.host}:{self.port}"
        )

    async def setup(self):
        """初始化 WebSocket 服务器"""
        logger.info(f"Protocol/WebSocket [{self.module_id}] 正在初始化...")
        self._is_initialized = True
        self._is_ready = True
        logger.info(f"Protocol/WebSocket [{self.module_id}] 初始化成功")

    async def start(self):
        """启动 WebSocket 服务器"""
        logger.info(
            f"Protocol/WebSocket [{self.module_id}] 启动服务器: "
            f"ws://{self.host}:{self.port}"
        )
        self.server = await websockets.serve(self._handle_client, self.host, self.port)
        logger.info(f"Protocol/WebSocket [{self.module_id}] 服务器已启动")
        await self.server.wait_closed()

    async def stop(self):
        """停止 WebSocket 服务器"""
        logger.info(f"Protocol/WebSocket [{self.module_id}] 正在停止服务器...")
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        logger.info(f"Protocol/WebSocket [{self.module_id}] 服务器已停止")

    async def close(self):
        """关闭协议，释放资源"""
        logger.info(f"Protocol/WebSocket [{self.module_id}] 正在关闭...")
        await self.stop()
        await self._cleanup_all_sessions()
        self._is_ready = False
        self._is_initialized = False
        logger.info(f"Protocol/WebSocket [{self.module_id}] 已关闭")
        await super().close()

    # ==================== 连接管理 ====================

    async def _handle_client(
        self,
        websocket: WebSocketServerProtocol,
        path: Optional[str] = ""
    ):
        """处理客户端连接"""
        tag_id, session_id = await self._register_client(websocket)
        if not session_id:
            return

        try:
            # 接收消息循环
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    # 音频数据 → 转发给 ConversationHandler
                    self._handle_audio(raw_message, session_id)
                else:
                    # 文本/事件消息 → 转发给 ConversationHandler
                    await self._handle_text_message(raw_message, session_id)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 连接错误 "
                f"(session: {session_id}): {e}"
            )

        finally:
            await self._cleanup_handler(websocket)

    async def _register_client(
        self,
        websocket: WebSocketServerProtocol
    ) -> Tuple[Optional[str], Optional[str]]:
        """注册新客户端连接"""
        try:
            # 接收注册消息
            register_data = await websocket.recv()
            stream_event = StreamEvent.model_validate_json(register_data)

            if stream_event.event_type != EventType.SYSTEM_CLIENT_SESSION_START:
                return None, None

            tag_id = stream_event.tag_id

            # 使用基类方法创建会话映射
            session_id = self.create_session(websocket, tag_id)

            # 通过 ChatEngine 创建 ConversationHandler
            await self.chat_engine.create_conversation_handler(
                session_id=session_id,
                tag_id=tag_id,
                send_callback=lambda event: self.send_event(session_id, event)
            )

            # 发送会话确认
            assignment_message = StreamEvent(
                event_type=EventType.SYSTEM_SERVER_SESSION_START,
                tag_id=tag_id,
                session_id=session_id
            )
            await websocket.send(assignment_message.model_dump_json())

            logger.info(
                f"Protocol/WebSocket [{self.module_id}] 客户端已注册: "
                f"tag={tag_id}, session={session_id}"
            )

            return tag_id, session_id

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 客户端注册失败: {e}"
            )
            return None, None

    async def _cleanup_handler(self, websocket: WebSocketServerProtocol):
        """清理会话"""
        # 使用基类方法移除会话映射
        session_id = self.remove_session_by_connection(websocket)
        if not session_id:
            return

        logger.info(
            f"Protocol/WebSocket [{self.module_id}] 清理会话: {session_id}"
        )

        # 通过 ChatEngine 销毁 ConversationHandler
        await self.chat_engine.destroy_conversation_handler(session_id)

    async def _cleanup_all_sessions(self):
        """清理所有会话"""
        logger.info(f"Protocol/WebSocket [{self.module_id}] 清理所有会话")

        # 获取所有 session_id
        session_ids = list(self.session_to_connection.keys())

        # 通过 ChatEngine 销毁所有 ConversationHandler
        for session_id in session_ids:
            await self.chat_engine.destroy_conversation_handler(session_id)

        # 使用基类方法清理映射
        self.clear_all_sessions()

    # ==================== 消息处理 ====================

    async def _handle_text_message(self, raw_message: str, session_id: str):
        """处理文本消息 → 转发给 ConversationHandler"""
        try:
            if not raw_message.strip().startswith('{'):
                return

            message_data = StreamEvent.model_validate_json(raw_message)
            handler = self.chat_engine.get_conversation_handler(session_id)

            if not handler:
                logger.warning(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"会话处理器不存在: {session_id}"
                )
                return

            # 根据事件类型分发
            if message_data.event_type == EventType.CLIENT_SPEECH_END:
                logger.debug(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"收到 CLIENT_SPEECH_END (session: {session_id})"
                )
                handler.handle_speech_end()

            elif message_data.event_type == EventType.STREAM_END:
                logger.debug(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"收到 STREAM_END (session: {session_id})"
                )
                handler.handle_speech_end()

            elif message_data.event_type == EventType.CLIENT_TEXT_INPUT:
                logger.debug(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"收到 CLIENT_TEXT_INPUT (session: {session_id})"
                )
                await handler.handle_text_input(message_data.event_data.text)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 消息解析失败: {e}"
            )

    def _handle_audio(self, audio_data: bytes, session_id: str):
        """处理音频数据 → 转发给 ConversationHandler"""
        handler = self.chat_engine.get_conversation_handler(session_id)

        if not handler:
            logger.warning(
                f"Protocol/WebSocket [{self.module_id}] "
                f"会话处理器不存在: {session_id}"
            )
            return

        handler.handle_audio(audio_data)

    # ==================== 协议特定方法 ====================

    async def send_message(
        self,
        connection: WebSocketServerProtocol,
        message: str
    ):
        """发送消息到 WebSocket 连接"""
        try:
            await connection.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.debug(
                f"Protocol/WebSocket [{self.module_id}] 连接已关闭"
            )
        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 发送消息失败: {e}"
            )
