import asyncio
import re
from typing import Optional, Dict, Callable, Awaitable

import constant
from models import StreamEvent, EventType, TextData
from modules import BaseLLM, BaseTTS, BaseVAD, BaseASR
from core.session_context import SessionContext
from core.session_manager import session_manager
from handlers import AudioInputHandler, TextInputHandler
from utils.logging_setup import logger


class ConversationHandler:
    """会话对话处理器

    职责:
    - 管理单个会话的对话流程
    - 处理音频 → ASR → LLM → TTS 完整链路
    - 管理打断逻辑和对话上下文
    - 与传输协议解耦（通过回调发送消息）
    """

    # 默认配置
    DEFAULT_SILENCE_TIMEOUT = 1.0
    DEFAULT_MAX_BUFFER_DURATION = 5.0
    SENTENCE_DELIMITER_PATTERN = re.compile(r'([，。！？；、,.!?;])')

    def __init__(
        self,
        session_id: str,
        tag_id: str,
        chat_engine: 'ChatEngine',
        send_callback: Callable[[StreamEvent], Awaitable[None]]
    ):
        self.session_id = session_id
        self.tag_id = tag_id
        self.chat_engine = chat_engine
        self.send_callback = send_callback

        # 对话状态
        self.turn_context = {
            'last_user_text': '',
            'was_interrupted': False
        }
        self.interrupt_flag = False

        # 业务组件
        self.audio_input: Optional[AudioInputHandler] = None
        self.text_input: Optional[TextInputHandler] = None

        logger.info(f"ConversationHandler 创建: session={session_id}")

    async def start(self):
        """启动对话处理器 - 创建 SessionContext 和输入处理器"""
        # 创建 SessionContext
        session_ctx = SessionContext(
            session_id=self.session_id,
            tag_id=self.tag_id,
            engine=self.chat_engine
        )
        await session_manager.create_session(session_ctx)

        # 创建 AudioInputHandler（会从 session_ctx 获取模块）
        self.audio_input = AudioInputHandler(
            session_context=session_ctx,
            result_callback=self._on_input_result,
            silence_timeout=self.DEFAULT_SILENCE_TIMEOUT,
            max_buffer_duration=self.DEFAULT_MAX_BUFFER_DURATION,
        )
        self.audio_input.start()

        # 创建 TextInputHandler
        self.text_input = TextInputHandler(
            session_context=session_ctx,
            result_callback=self._on_input_result
        )

        logger.info(f"ConversationHandler 启动完成: session={self.session_id}")

    async def stop(self):
        """停止对话处理器 - 清理资源"""
        logger.info(f"ConversationHandler 正在停止: session={self.session_id}")

        # 停止 AudioInputHandler
        if self.audio_input:
            self.audio_input.stop()
            self.audio_input = None

        # 清理状态
        self.turn_context.clear()

        logger.info(f"ConversationHandler 已停止: session={self.session_id}")

    # ==================== 消息处理 ====================

    async def handle_audio(self, audio_data: bytes):
        """处理音频数据"""
        # 音频到来 = 可能的打断
        if not self.interrupt_flag:
            self.interrupt_flag = True
            self.turn_context['was_interrupted'] = True
            logger.debug(f"ConversationHandler 检测到打断: session={self.session_id}")

        # 传递给 AudioInputHandler
        if self.audio_input:
            await self.audio_input.process_chunk(audio_data)

    async def handle_speech_end(self):
        """处理语音结束信号"""
        if self.audio_input:
            self.audio_input.signal_client_speech_end()

    async def handle_text_input(self, text: str):
        """处理文本输入（不打断）"""
        if self.text_input:
            await self.text_input.process_text(text)

    async def _on_input_result(
        self,
        input_event: StreamEvent,
        metadata: Optional[Dict]
    ):
        """输入结果回调 - 处理 ASR 或文本输入的结果"""
        text_data: TextData = input_event.event_data

        if not text_data.is_final:
            return

        # 空结果，重置打断标志
        if not text_data.text:
            logger.debug(f"ConversationHandler 输入结果为空: session={self.session_id}")
            self.turn_context['was_interrupted'] = False
            return

        # 处理打断场景（仅音频输入有打断逻辑）
        if self.turn_context['was_interrupted']:
            # 拼接上一轮和本轮文本
            combined_text = f"{self.turn_context.get('last_user_text', '')} {text_data.text}".strip()
            logger.info(
                f"ConversationHandler 拼接打断文本 (session: {self.session_id}): "
                f"'{combined_text}'"
            )
            user_text = combined_text
        else:
            user_text = text_data.text

        # 更新上下文
        self.turn_context['last_user_text'] = user_text
        self.turn_context['was_interrupted'] = False

        # 重置打断标志
        self.interrupt_flag = False

        # 触发对话
        await self._trigger_conversation(user_text)

    # ==================== 对话流程 ====================

    async def _trigger_conversation(self, user_text: str):
        """触发对话流程: LLM → TTS"""
        # 从当前会话获取模块
        session_ctx = await session_manager.get_session(self.session_id)
        llm_module: BaseLLM = session_ctx.get_module("llm")
        tts_module: BaseTTS = session_ctx.get_module("tts")

        if not llm_module:
            logger.error(f"ConversationHandler LLM 模块未找到: session={self.session_id}")
            return

        llm_input = TextData(text=user_text, is_final=True)

        if tts_module:
            await self._process_with_tts(llm_input, llm_module, tts_module)
        else:
            await self._process_text_only(llm_input, llm_module)

    async def _process_with_tts(
        self,
        llm_input: TextData,
        llm_module: BaseLLM,
        tts_module: BaseTTS
    ):
        """处理 LLM 输出并合成语音"""
        buffer = ""

        # 修复: BaseLLM 的方法是 chat_stream(text, session_id)
        async for text_chunk in llm_module.chat_stream(llm_input, self.session_id):
            # chat_stream 返回 AsyncGenerator[TextData, None]
            content = text_chunk.text if hasattr(text_chunk, 'text') else text_chunk
            # 检查打断
            if self.interrupt_flag:
                logger.info(f"ConversationHandler 对话被打断: session={self.session_id}")
                break

            if content:
                buffer += content
                match = self.SENTENCE_DELIMITER_PATTERN.search(buffer)

                if match:
                    sentence = buffer[:match.end()]
                    buffer = buffer[match.end():]
                    asyncio.create_task(
                        self._send_sentence(sentence, tts_module, is_final=False)
                    )

        # 发送剩余文本
        if buffer and not self.interrupt_flag:
            asyncio.create_task(
                self._send_sentence(buffer, tts_module, is_final=True)
            )

    async def _process_text_only(
        self,
        llm_input: TextData,
        llm_module: BaseLLM
    ):
        """只处理 LLM 输出（无 TTS）"""
        # 修复: BaseLLM 的方法是 chat_stream(text, session_id)
        async for text_chunk in llm_module.chat_stream(llm_input, self.session_id):
            content = text_chunk.text if hasattr(text_chunk, 'text') else text_chunk
            if self.interrupt_flag:
                break

            if content:
                text_event = StreamEvent(
                    event_type=EventType.SERVER_TEXT_RESPONSE,
                    event_data=TextData(text=content, is_final=False),
                    session_id=self.session_id
                )
                await self.send_callback(text_event)

        # 发送最终标记
        if not self.interrupt_flag:
            final_event = StreamEvent(
                event_type=EventType.SERVER_TEXT_RESPONSE,
                event_data=TextData(text="", is_final=True),
                session_id=self.session_id
            )
            await self.send_callback(final_event)

    async def _send_sentence(
        self,
        sentence: str,
        tts_module: BaseTTS,
        is_final: bool = False
    ):
        """发送句子（文本 + 音频）"""
        if self.interrupt_flag:
            return

        # 发送文本
        text_event = StreamEvent(
            event_type=EventType.SERVER_TEXT_RESPONSE,
            event_data=TextData(text=sentence, is_final=is_final),
            session_id=self.session_id
        )
        await self.send_callback(text_event)

        # 修复: BaseTTS.synthesize_stream 返回 AsyncGenerator[AudioData, None]
        # 需要遍历音频流并发送每个音频块
        async for audio_chunk in tts_module.synthesize_stream(TextData(text=sentence)):
            if self.interrupt_flag:
                break

            if audio_chunk and audio_chunk.data:
                audio_event = StreamEvent(
                    event_type=EventType.SERVER_AUDIO_RESPONSE,
                    event_data=audio_chunk,
                    session_id=self.session_id
                )
                await self.send_callback(audio_event)
