import asyncio
import base64
import json
import re
import uuid
from typing import Optional, Dict, Tuple

import websockets
from websockets.server import WebSocketServerProtocol

import constant
from core.session_context import SessionContext
from core.session_manager import session_manager
from data_models import StreamEvent, EventType, TextData, AudioData
from modules import BaseLLM, BaseTTS, BaseVAD, BaseASR
from modules.base_protocol import BaseProtocol
from service.AudioConsumer import AudioConsumer
from utils.logging_setup import logger


class WebSocketProtocolAdapter(BaseProtocol):
    """WebSocket 协议适配器

    管理 WebSocket 连接、会话和消息路由。
    """

    # 默认配置常量
    DEFAULT_SILENCE_TIMEOUT = 1.0
    DEFAULT_MAX_BUFFER_DURATION = 5.0
    SENTENCE_DELIMITER_PATTERN = re.compile(r'([，。！？；、,.!?;])')

    def __init__(self, module_id: str, config: Dict):
        super().__init__(module_id, config)

        # WebSocket 服务器
        self.server: Optional[websockets.WebSocketServer] = None

        # 会话映射
        self.tag_to_session: Dict[str, str] = {}
        self.session_to_websocket: Dict[str, WebSocketServerProtocol] = {}
        self.websocket_to_session: Dict[WebSocketServerProtocol, str] = {}

        # 会话资源
        self.audio_consumers: Dict[str, AudioConsumer] = {}
        self.session_interrupt_flags: Dict[str, bool] = {}
        self.session_turn_context: Dict[str, Dict[str, any]] = {}

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
        self._cleanup_all_sessions()
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

        metadata = {"session_id": session_id, "tag_id": tag_id}

        try:
            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    self._handle_audio(raw_message, metadata)
                else:
                    await self._handle_text_message(raw_message, metadata)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 连接错误 "
                f"(session: {session_id}): {e}"
            )

        finally:
            self._cleanup_session(websocket)

    async def _register_client(
        self,
        websocket: WebSocketServerProtocol
    ) -> Tuple[Optional[str], Optional[str]]:
        """注册新客户端连接"""
        try:
            register_data = await websocket.recv()
            stream_event = StreamEvent.model_validate_json(register_data)

            if stream_event.event_type != EventType.SYSTEM_CLIENT_SESSION_START:
                return None, None

            tag_id = stream_event.tag_id
            session_id = str(uuid.uuid4())

            # 建立映射关系
            self.tag_to_session[tag_id] = session_id
            self.session_to_websocket[session_id] = websocket
            self.websocket_to_session[websocket] = session_id

            # 初始化会话资源
            self._init_session_resources(session_id, tag_id)

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

    def _init_session_resources(self, session_id: str, tag_id: str):
        """初始化会话资源"""
        # 初始化对话上下文
        self.session_turn_context[session_id] = {
            'last_user_text': '',
            'was_interrupted': False
        }
        self.session_interrupt_flags[session_id] = False

        # 创建会话上下文
        user_context = SessionContext()
        user_context.session_id = session_id
        user_context.tag_id = tag_id
        session_manager.create_session(user_context)

        # 获取模块
        context = session_manager.get_session(constant.CHAT_ENGINE_NAME)
        vad_module: BaseVAD = context.global_module_manager.get_module("vad")
        asr_module: BaseASR = context.global_module_manager.get_module("asr")

        # 创建音频消费者
        consumer = AudioConsumer(
            session_context=user_context,
            vad_module=vad_module,
            asr_module=asr_module,
            asr_result_callback=self._handle_asr_result,
            silence_timeout=self.DEFAULT_SILENCE_TIMEOUT,
            max_buffer_duration=self.DEFAULT_MAX_BUFFER_DURATION,
        )
        consumer.start()
        self.audio_consumers[session_id] = consumer

    def _cleanup_session(self, websocket: WebSocketServerProtocol):
        """清理会话资源"""
        session_id = self.websocket_to_session.pop(websocket, None)
        if not session_id:
            return

        logger.info(
            f"Protocol/WebSocket [{self.module_id}] 清理会话: {session_id}"
        )

        # 停止音频消费者
        if session_id in self.audio_consumers:
            self.audio_consumers[session_id].stop()
            del self.audio_consumers[session_id]

        # 清理各种映射
        self.session_interrupt_flags.pop(session_id, None)
        self.session_to_websocket.pop(session_id, None)
        self.session_turn_context.pop(session_id, None)

    def _cleanup_all_sessions(self):
        """清理所有会话资源"""
        logger.info(f"Protocol/WebSocket [{self.module_id}] 清理所有会话")

        for consumer in self.audio_consumers.values():
            consumer.stop()

        self.audio_consumers.clear()
        self.session_interrupt_flags.clear()
        self.session_to_websocket.clear()
        self.websocket_to_session.clear()
        self.session_turn_context.clear()
        self.tag_to_session.clear()

    # ==================== 消息处理 ====================

    async def _handle_text_message(self, raw_message: str, metadata: Dict):
        """处理文本消息"""
        try:
            if not raw_message.strip().startswith('{'):
                return

            message_data = StreamEvent.model_validate_json(raw_message)
            await self._handle_stream_event(message_data, metadata)

        except Exception as e:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] 消息解析失败: {e}"
            )

    async def _handle_stream_event(
        self,
        message_data: StreamEvent,
        metadata: Dict
    ):
        """处理流事件"""
        session_id = metadata["session_id"]
        event_type = message_data.event_type

        if event_type == EventType.CLIENT_SPEECH_END:
            logger.debug(
                f"Protocol/WebSocket [{self.module_id}] "
                f"收到 CLIENT_SPEECH_END (session: {session_id})"
            )
            if session_id in self.audio_consumers:
                self.audio_consumers[session_id].signal_client_speech_end()

        elif event_type == EventType.STREAM_END:
            logger.debug(
                f"Protocol/WebSocket [{self.module_id}] "
                f"收到 STREAM_END (session: {session_id})"
            )
            if session_id in self.audio_consumers:
                self.audio_consumers[session_id].signal_client_speech_end()

        elif event_type == EventType.CLIENT_TEXT_INPUT:
            # 文本输入不算打断，直接重置上下文
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['last_user_text'] = \
                    message_data.event_data.text
                self.session_turn_context[session_id]['was_interrupted'] = False

            await self._trigger_llm_and_tts(message_data.event_data, metadata)

    def _handle_audio(self, audio_data: bytes, metadata: Dict):
        """处理音频数据"""
        session_id = metadata["session_id"]

        # 检测打断
        if not self.session_interrupt_flags.get(session_id, False):
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['was_interrupted'] = True
                logger.debug(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"检测到打断 (session: {session_id})"
                )

        self.session_interrupt_flags[session_id] = True

        # 传递给音频消费者
        if session_id in self.audio_consumers:
            self.audio_consumers[session_id].process_chunk(audio_data)

    async def _handle_asr_result(
        self,
        asr_event: StreamEvent,
        metadata: Optional[Dict]
    ):
        """处理 ASR 识别结果"""
        session_id = metadata["session_id"]
        text_data: TextData = asr_event.event_data

        if not text_data.is_final:
            return

        # 空结果，重置打断标志
        if not text_data.text:
            logger.debug(
                f"Protocol/WebSocket [{self.module_id}] "
                f"ASR 结果为空 (session: {session_id})"
            )
            if session_id in self.session_turn_context:
                self.session_turn_context[session_id]['was_interrupted'] = False
            return

        # 处理打断场景
        turn_context = self.session_turn_context.get(session_id)
        if turn_context and turn_context['was_interrupted']:
            # 拼接上一轮和本轮文本
            combined_text = f"{turn_context.get('last_user_text', '')} {text_data.text}".strip()
            logger.info(
                f"Protocol/WebSocket [{self.module_id}] "
                f"拼接打断文本 (session: {session_id}): '{combined_text}'"
            )
            llm_input_text = combined_text
        else:
            llm_input_text = text_data.text

        # 更新上下文
        if turn_context:
            turn_context['last_user_text'] = llm_input_text
            turn_context['was_interrupted'] = False

        # 重置打断标志，准备播放新回答
        self.session_interrupt_flags[session_id] = False

        # 触发 LLM 和 TTS
        llm_input_data = TextData(text=llm_input_text, is_final=True)
        await self._trigger_llm_and_tts(llm_input_data, metadata)

    # ==================== LLM & TTS ====================

    async def _trigger_llm_and_tts(
        self,
        text_data: TextData,
        metadata: Dict
    ):
        """触发 LLM 对话和 TTS 合成"""
        session_id = metadata["session_id"]
        websocket = self.session_to_websocket.get(session_id)

        if not websocket:
            return

        # 获取模块
        context = session_manager.get_session(constant.CHAT_ENGINE_NAME)
        llm_module: BaseLLM = context.global_module_manager.get_module("llm")
        tts_module: BaseTTS = context.global_module_manager.get_module("tts")

        if not llm_module:
            logger.error(
                f"Protocol/WebSocket [{self.module_id}] LLM 模块未找到"
            )
            return

        # 流式接收 LLM 输出，按句子切分
        buffer = ""

        async for content in llm_module.stream_chat_response(text_data):
            # 检查打断
            if self.session_interrupt_flags.get(session_id, False):
                logger.info(
                    f"Protocol/WebSocket [{self.module_id}] "
                    f"TTS 被打断 (session: {session_id})"
                )
                break

            if content:
                buffer += content
                match = self.SENTENCE_DELIMITER_PATTERN.search(buffer)

                if match:
                    sentence = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    asyncio.create_task(
                        self._process_and_send_tts(
                            sentence, session_id, tts_module
                        )
                    )

        # 发送剩余文本
        if buffer and not self.session_interrupt_flags.get(session_id, False):
            asyncio.create_task(
                self._process_and_send_tts(
                    buffer, session_id, tts_module, is_final=True
                )
            )

    async def _process_and_send_tts(
        self,
        sentence: str,
        session_id: str,
        tts_module: BaseTTS,
        is_final: bool = False
    ):
        """处理并发送 TTS 结果"""
        websocket = self.session_to_websocket.get(session_id)

        if not websocket or self.session_interrupt_flags.get(session_id, False):
            return

        # 发送文本响应
        text_event = StreamEvent(
            event_type=EventType.SERVER_TEXT_RESPONSE,
            event_data=TextData(text=sentence, is_final=is_final),
            session_id=session_id
        )
        await self._send_to_client(websocket, text_event.model_dump_json())

        # 生成并发送音频
        audio_data = await tts_module.text_to_speech_block(
            TextData(text=sentence)
        )

        if audio_data and audio_data.data:
            if not self.session_interrupt_flags.get(session_id, False):
                audio_event = StreamEvent(
                    event_type=EventType.SERVER_AUDIO_RESPONSE,
                    event_data=audio_data,
                    session_id=session_id
                )
                await self._send_to_client(
                    websocket,
                    self._serialize_audio_event(audio_event)
                )

    # ==================== 工具方法 ====================

    @staticmethod
    def _serialize_audio_event(event: StreamEvent) -> str:
        """序列化音频事件 (Base64 编码)"""
        event_dict = event.model_dump()
        raw_data_bytes = event_dict['event_data']['data']
        base64_encoded_data = base64.b64encode(raw_data_bytes).decode('utf-8')
        event_dict['event_data']['data'] = base64_encoded_data
        return json.dumps(event_dict)

    @staticmethod
    async def _send_to_client(
        websocket: WebSocketServerProtocol,
        message: str
    ):
        """发送消息到客户端"""
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Protocol/WebSocket 发送消息失败: {e}")
