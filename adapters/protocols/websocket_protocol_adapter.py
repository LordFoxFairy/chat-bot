from typing import Optional, Dict

import websockets
from websockets.server import WebSocketServerProtocol

from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger


class WebSocketProtocolAdapter(BaseProtocol[WebSocketServerProtocol]):
    """WebSocket 协议适配器

    职责:
    - 管理 WebSocket 连接
    - 接收/发送消息
    - 调用基类通用方法处理业务逻辑
    - 纯传输层实现
    """

    def __init__(self, module_id: str, config: Dict, chat_engine: 'ChatEngine'):
        super().__init__(module_id, config, chat_engine)

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
        """处理客户端连接 - 调用基类通用方法"""
        try:
            # 接收消息循环
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    # 音频数据 → 基类通用方法
                    await self.handle_audio_message(websocket, raw_message)
                else:
                    # 文本/事件消息 → 基类通用方法
                    await self.handle_text_message(websocket, raw_message)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 连接错误: {e}"
            )

        finally:
            # 处理断开 → 基类通用方法
            await self.handle_disconnect(websocket)


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
