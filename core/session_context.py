# core_framework/session_context.py
import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional

import websockets  # 导入 websockets 以访问 State 枚举

logger = logging.getLogger(__name__)


class SessionContext:
    """
    代表一个独立的用户会话上下文。
    存储与单个用户会话相关的数据，包括其配置和关联的 WebSocket 连接。
    """

    def __init__(self,
                 session_id: str,
                 websocket: Any,  # WebSocket 连接对象
                 global_config: Dict[str, Any],  # 全局应用配置
                 user_specific_config: Optional[Dict[str, Any]] = None  # 用户特定配置
                 ):
        """
        初始化会话上下文。

        Args:
            session_id (str): 此会话的唯一标识符 (通常是客户端提供的 UUID)。
            websocket (Any): 与此会话关联的 WebSocket 连接对象。
                             期望是 websockets.WebSocketServerProtocol 的实例。
            global_config (Dict[str, Any]): 加载后的全局应用配置。
            user_specific_config (Optional[Dict[str, Any]]): 用户的特定配置，用于覆盖全局配置。
        """
        self.session_id = session_id
        self.websocket = websocket
        self.global_config = global_config
        self.user_specific_config_overrides = user_specific_config if user_specific_config else {}

        self.session_config = self._merge_configs()

        self.last_activity_time: float = time.time()
        self.dialog_history: list = []
        self.current_processing_task: Optional[asyncio.Task] = None

        logger.info(
            f"会话上下文 '{self.session_id}' 已创建并关联到 WebSocket: {self.websocket.remote_address if hasattr(self.websocket, 'remote_address') else '未知地址'}")

    def _merge_configs(self) -> Dict[str, Any]:
        """
        合并全局配置和用户特定配置。
        用户特定配置会覆盖全局配置中的相应部分。
        """
        merged_config = json.loads(json.dumps(self.global_config))  # 深拷贝

        def _deep_update(source_dict: Dict, updates: Dict):
            for key, value in updates.items():
                if isinstance(value, dict) and key in source_dict and isinstance(source_dict[key], dict):
                    _deep_update(source_dict[key], value)
                else:
                    source_dict[key] = value
            return source_dict

        merged_config = _deep_update(merged_config, self.user_specific_config_overrides)
        logger.debug(f"会话 '{self.session_id}' 的合并配置 (部分): {str(merged_config)[:500]}...")
        return merged_config

    def update_activity_time(self):
        """更新会话的最后活动时间。"""
        self.last_activity_time = time.time()

    async def send_message_to_client(self, message_data: Dict[str, Any]):
        """
        通过关联的 WebSocket 连接向客户端发送 JSON 消息。
        """
        # 修正：使用 websocket.state 检查连接状态
        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.State.OPEN:
            try:
                await self.websocket.send(json.dumps(message_data))
                logger.debug(f"[{self.session_id}] 已发送消息给客户端: {str(message_data)[:100]}")
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[{self.session_id}] 发送消息失败：WebSocket 连接已关闭。")
            except Exception as e:
                logger.error(f"[{self.session_id}] 发送消息给客户端失败: {e}", exc_info=True)
        else:
            logger.warning(
                f"[{self.session_id}] 尝试通过已关闭、不存在或状态未知的 WebSocket 发送消息。当前状态: {getattr(self.websocket, 'state', '未知')}")

    async def send_audio_to_client(self, audio_bytes: bytes):
        """
        通过关联的 WebSocket 连接向客户端发送二进制音频数据。
        """
        # 修正：使用 websocket.state 检查连接状态
        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.State.OPEN:
            try:
                await self.websocket.send(audio_bytes)
                logger.debug(f"[{self.session_id}] 已发送 {len(audio_bytes)} 字节的音频数据给客户端。")
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"[{self.session_id}] 发送音频失败：WebSocket 连接已关闭。")
            except Exception as e:
                logger.error(f"[{self.session_id}] 发送音频给客户端失败: {e}", exc_info=True)
        else:
            logger.warning(
                f"[{self.session_id}] 尝试通过已关闭、不存在或状态未知的 WebSocket 发送音频。当前状态: {getattr(self.websocket, 'state', '未知')}")

    async def close(self, reason: str = "会话关闭"):
        """
        清理此会话上下文的资源。
        主要是关闭关联的 WebSocket 连接（如果仍然打开）。
        """
        logger.info(f"正在关闭会话上下文 '{self.session_id}'，原因: {reason}...")
        # 修正：使用 websocket.state 检查连接状态
        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.State.OPEN:
            try:
                # 尝试发送一个会话结束的消息给客户端
                # 注意：如果连接在发送此消息前关闭，这里也可能抛出 ConnectionClosed
                await self.websocket.send(
                    json.dumps({"type": "session_ended", "session_id": self.session_id, "reason": reason}))
                await self.websocket.close(code=1000, reason=reason)
                logger.info(f"[{self.session_id}] 关联的 WebSocket 连接已关闭。")
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"[{self.session_id}] WebSocket 连接在尝试发送 session_ended 或关闭时已关闭。")
            except Exception as e_ws_close:
                logger.warning(f"[{self.session_id}] 关闭关联的 WebSocket 时发生错误: {e_ws_close}")

        self.websocket = None  # 清除引用
        logger.info(f"会话上下文 '{self.session_id}' 已关闭。")
