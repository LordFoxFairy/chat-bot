# chat-bot/core/session_context.py
import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Deque
from collections import deque

import websockets

logger = logging.getLogger(__name__)


class SessionContext:
    """
    代表一个独立的用户会话上下文。
    存储与单个用户会话相关的数据，包括其配置、WebSocket连接、激活状态以及VAD处理状态。
    """

    def __init__(self,
                 session_id: str,
                 websocket: Any,
                 global_config: Dict[str, Any],
                 user_specific_config: Optional[Dict[str, Any]] = None
                 ):
        self._max_speech_segment_ms_vad = 1200
        self.session_id = session_id
        self.websocket = websocket
        self.global_config = global_config
        self.user_specific_config_overrides = user_specific_config if user_specific_config else {}
        self.session_config = self._merge_configs()

        activation_settings = self.session_config.get("activation_settings", {})
        self.enable_prompt_activation: bool = activation_settings.get("enable_prompt_activation", False)
        self.activation_keywords: List[str] = activation_settings.get("activation_keywords", ["你好小爱", "你好助手"])
        self.activation_timeout_seconds: int = activation_settings.get("activation_timeout_seconds", 30)
        self.activation_reply: str = activation_settings.get("activation_reply", "我在，有什么可以帮您的吗？")
        self.deactivation_reply: str = activation_settings.get("deactivation_reply", "再见，期待下次为您服务。")
        self.prompt_if_not_activated: Optional[str] = activation_settings.get("prompt_if_not_activated", None)
        self.is_active: bool = not self.enable_prompt_activation

        self.last_websocket_activity_time: float = time.time()
        self.last_interaction_time: float = time.time()
        self.dialog_history: list = []

        # 并发任务引用
        self.current_processing_task: Optional[
            asyncio.Task] = None  # 指向主要的 _process_message_for_session 或 _process_active_conversation 中的某个关键任务
        self.tts_consumer_task: Optional[asyncio.Task] = None
        self.client_text_sender_task: Optional[asyncio.Task] = None
        self.llm_producer_task: Optional[asyncio.Task] = None

        # VAD处理相关状态属性
        self._raw_audio_buffer_vad: Deque[bytes] = deque()
        self._speech_utterance_buffer_vad: Deque[bytes] = deque()
        self._is_currently_speaking_vad: bool = False
        self._silence_since_last_speech_vad: Optional[float] = None
        self._vad_window_size_bytes_vad: Optional[int] = None
        self._min_silence_for_eos_ms_vad: Optional[int] = None
        self._asr_triggered_for_utterance_vad: bool = False
        self._last_vad_judgement_is_speech: bool = False
        self._vad_produced_asr_result_this_stream: bool = False
        self._client_text_queue_ended: bool = False  # 用于标记客户端文本队列是否已发送结束信号

        # session_manager 的引用不应该在这里，它由 ChatEngine 管理，并传递给需要的模块（如BaseLLM）
        # self.session_manager = None

        logger.info(
            f"会话上下文 '{self.session_id}' 已创建。启用提示词激活: {self.enable_prompt_activation}."
            f"初始激活状态: {self.is_active}. "
            f"关联到 WebSocket: {self.websocket.remote_address if hasattr(self.websocket, 'remote_address') else '未知地址'}")
        if self.enable_prompt_activation:
            logger.info(
                f"[{self.session_id}] 激活关键词: {self.activation_keywords}, 超时: {self.activation_timeout_seconds}s")

    def _merge_configs(self) -> Dict[str, Any]:
        merged_config = json.loads(json.dumps(self.global_config))

        def _deep_update(source_dict: Dict, updates: Dict):
            for key, value in updates.items():
                if isinstance(value, dict) and key in source_dict and isinstance(source_dict[key], dict):
                    _deep_update(source_dict[key], value)
                else:
                    source_dict[key] = value
            return source_dict

        merged_config = _deep_update(merged_config, self.user_specific_config_overrides)
        return merged_config

    def update_websocket_activity_time(self):
        self.last_websocket_activity_time = time.time()

    def update_interaction_time(self):
        self.last_interaction_time = time.time()
        logger.debug(f"[{self.session_id}] 最后交互时间已更新。")

    async def send_message_to_client(self, message_data: Dict[str, Any]):
        # 使用 websockets.protocol.State.OPEN 进行比较
        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.protocol.State.OPEN:  # type: ignore
            try:
                await self.websocket.send(json.dumps(message_data))
            except websockets.exceptions.ConnectionClosed:  # type: ignore
                logger.warning(f"[{self.session_id}] 发送消息失败：WebSocket 连接已关闭。")
            except Exception as e:
                logger.error(f"[{self.session_id}] 发送消息给客户端失败: {e}", exc_info=True)
        else:
            logger.warning(
                f"[{self.session_id}] 尝试通过已关闭的 WebSocket 发送消息。当前状态: {getattr(self.websocket, 'state', '未知')}")

    async def send_audio_to_client(self, audio_bytes: bytes):
        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.protocol.State.OPEN:  # type: ignore
            try:
                await self.websocket.send(audio_bytes)
            except websockets.exceptions.ConnectionClosed:  # type: ignore
                logger.warning(f"[{self.session_id}] 发送音频失败：WebSocket 连接已关闭。")
            except Exception as e:
                logger.error(f"[{self.session_id}] 发送音频给客户端失败: {e}", exc_info=True)
        else:
            logger.warning(
                f"[{self.session_id}] 尝试通过已关闭的 WebSocket 发送音频。当前状态: {getattr(self.websocket, 'state', '未知')}")

    async def close(self, reason: str = "会话关闭"):
        logger.info(f"正在关闭会话上下文 '{self.session_id}'，原因: {reason}...")

        tasks_to_cancel = [
            self.current_processing_task,
            self.tts_consumer_task,
            self.client_text_sender_task,
            self.llm_producer_task
        ]
        for task in tasks_to_cancel:
            if task and not task.done():
                task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
                logger.debug(f"[{self.session_id}] 正在取消任务: {task_name}")
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)  # 给任务一点时间响应取消
                except asyncio.CancelledError:
                    logger.debug(f"[{self.session_id}] 任务 {task_name} 已成功取消。")
                except asyncio.TimeoutError:  # pragma: no cover
                    logger.warning(f"[{self.session_id}] 取消任务 {task_name} 超时。")
                except Exception as e_task_cancel:  # pragma: no cover
                    logger.error(f"[{self.session_id}] 取消任务 {task_name} 时发生错误: {e_task_cancel}", exc_info=True)

        self.current_processing_task = None
        self.tts_consumer_task = None
        self.client_text_sender_task = None
        self.llm_producer_task = None

        if self.websocket and hasattr(self.websocket, 'state') and \
                self.websocket.state == websockets.protocol.State.OPEN:  # type: ignore
            try:
                await self.websocket.send(
                    json.dumps({"type": "session_ended", "session_id": self.session_id, "reason": reason}))
                await self.websocket.close(code=1000, reason=reason)
            except websockets.exceptions.ConnectionClosed:  # type: ignore
                logger.info(f"[{self.session_id}] WebSocket 连接在尝试发送 session_ended 或关闭时已关闭。")
            except Exception as e_ws_close:  # pragma: no cover
                logger.warning(f"[{self.session_id}] 关闭关联的 WebSocket 时发生错误: {e_ws_close}")

        self.websocket = None
        logger.info(f"会话上下文 '{self.session_id}' 已关闭。")

    @property
    def vad_produced_asr_result_this_stream(self):
        return self._vad_produced_asr_result_this_stream
