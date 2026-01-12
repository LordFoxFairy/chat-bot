from typing import Optional, Dict

import websockets
from websockets.server import WebSocketServerProtocol

from data_models import StreamEvent, EventType, TextData
from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger


class WebSocketProtocolAdapter(BaseProtocol[WebSocketServerProtocol]):
    """WebSocket 协议适配器

    职责:
    - 管理 WebSocket 连接
    - 接收/发送消息
    - 解析协议消息并路由到 ChatEngine
    - 依赖 ChatEngine（单向依赖）
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
        self.clear_all_sessions()
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
        try:
            # 接收消息循环
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    # 音频数据
                    await self._handle_audio_message(websocket, raw_message)
                else:
                    # 文本/事件消息
                    await self._handle_text_message(websocket, raw_message)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 连接错误: {e}"
            )

        finally:
            # 处理断开
            await self._handle_disconnect(websocket)

    async def _handle_text_message(self, websocket: WebSocketServerProtocol, raw_message: str):
        """处理文本消息"""
        try:
            if not raw_message.strip().startswith('{'):
                return

            stream_event = StreamEvent.model_validate_json(raw_message)

            # 判断是否为注册消息
            if stream_event.event_type == EventType.SYSTEM_CLIENT_SESSION_START:
                await self._handle_register(websocket, stream_event)
            else:
                # 路由到 ConversationHandler
                await self._route_message(websocket, stream_event)

        except Exception as e:
            logger.error(f"Protocol/WebSocket [{self.module_id}] 消息处理失败: {e}")

    async def _handle_register(self, websocket: WebSocketServerProtocol, stream_event: StreamEvent):
        """处理注册消息"""
        tag_id = stream_event.tag_id

        # 创建会话映射
        session_id = self.create_session(websocket, tag_id)

        # 调用 ChatEngine 创建 ConversationHandler
        await self.chat_engine.create_conversation_handler(
            session_id=session_id,
            tag_id=tag_id,
            send_callback=lambda event: self.send_event(session_id, event)
        )

        # 发送会话确认
        response = StreamEvent(
            event_type=EventType.SYSTEM_SERVER_SESSION_START,
            tag_id=tag_id,
            session_id=session_id
        )
        await self.send_event(session_id, response)

        logger.info(f"Protocol/WebSocket [{self.module_id}] 客户端已注册: tag={tag_id}, session={session_id}")

    async def _route_message(self, websocket: WebSocketServerProtocol, stream_event: StreamEvent):
        """路由消息到 ConversationHandler"""
        session_id = self.get_session_id(websocket)
        if not session_id:
            logger.warning(f"Protocol/WebSocket [{self.module_id}] 未找到会话映射")
            return

        handler = self.chat_engine.get_conversation_handler(session_id)
        if not handler:
            logger.warning(f"Protocol/WebSocket [{self.module_id}] 会话处理器不存在: {session_id}")
            return

        # 根据事件类型分发
        if stream_event.event_type == EventType.CLIENT_SPEECH_END:
            handler.handle_speech_end()
        elif stream_event.event_type == EventType.STREAM_END:
            handler.handle_speech_end()
        elif stream_event.event_type == EventType.CLIENT_TEXT_INPUT:
            text_data: TextData = stream_event.event_data
            await handler.handle_text_input(text_data.text)

    async def _handle_audio_message(self, websocket: WebSocketServerProtocol, audio_data: bytes):
        """处理音频消息"""
        session_id = self.get_session_id(websocket)
        if not session_id:
            return

        handler = self.chat_engine.get_conversation_handler(session_id)
        if handler:
            handler.handle_audio(audio_data)

    async def _handle_disconnect(self, websocket: WebSocketServerProtocol):
        """处理断开连接"""
        session_id = self.remove_session_by_connection(websocket)
        if session_id:
            logger.info(f"Protocol/WebSocket [{self.module_id}] 连接断开: session={session_id}")
            await self.chat_engine.destroy_conversation_handler(session_id)


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
