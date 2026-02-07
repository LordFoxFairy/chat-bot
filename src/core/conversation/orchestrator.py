import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Set

from src.core.models import StreamEvent, EventType, TextData
from src.core.interfaces import BaseLLM, BaseTTS
from src.core.session.session_context import SessionContext
from src.core.session.session_manager import SessionManager
from src.core.conversation.interrupt_manager import InterruptManager
from src.core.conversation.sentence_splitter import SentenceSplitter
from src.core.input.audio_handler import AudioInputHandler
from src.core.input.text_handler import TextInputHandler
from src.utils.logging_setup import logger


class ConversationOrchestrator:
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

    def __init__(
        self,
        session_id: str,
        tag_id: str,
        session_context: SessionContext,
        session_manager: SessionManager,
        send_callback: Callable[[StreamEvent], Awaitable[None]]
    ):
        self.session_id = session_id
        self.tag_id = tag_id
        self.session_context = session_context
        self.session_manager = session_manager
        self.send_callback = send_callback

        # 对话状态
        self.turn_context: Dict[str, Any] = {
            'last_user_text': '',
        }

        # 使用新的管理器
        self.interrupt_manager = InterruptManager(session_id)
        self.sentence_splitter = SentenceSplitter()

        # 业务组件
        self.audio_input: Optional[AudioInputHandler] = None
        self.text_input: Optional[TextInputHandler] = None

        # 保存后台任务引用，防止被垃圾回收
        self._pending_tasks: Set[asyncio.Task] = set()

        logger.info(f"ConversationOrchestrator 创建: session={session_id}")

    async def start(self):
        """启动对话处理器 - 创建输入处理器"""
        # SessionContext 已经在外部创建并传入
        # 注册到 SessionManager
        await self.session_manager.create_session(self.session_context)

        # 创建 AudioInputHandler（会从 session_ctx 获取模块）
        self.audio_input = AudioInputHandler(
            session_context=self.session_context,
            result_callback=self._on_input_result,
            silence_timeout=self.DEFAULT_SILENCE_TIMEOUT,
            max_buffer_duration=self.DEFAULT_MAX_BUFFER_DURATION,
        )
        self.audio_input.start()

        # 创建 TextInputHandler
        self.text_input = TextInputHandler(
            session_context=self.session_context,
            result_callback=self._on_input_result
        )

        logger.info(f"ConversationOrchestrator 启动完成: session={self.session_id}")

    async def stop(self):
        """停止对话处理器 - 清理资源"""
        logger.info(f"ConversationOrchestrator 正在停止: session={self.session_id}")

        # 取消所有待处理的任务
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        # 等待所有任务完成或被取消
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
        self._pending_tasks.clear()

        # 停止 AudioInputHandler
        if self.audio_input:
            await self.audio_input.stop()
            self.audio_input = None

        # 清理 TextInputHandler
        if self.text_input:
            self.text_input = None

        # 清理状态
        self.turn_context.clear()

        logger.info(f"ConversationOrchestrator 已停止: session={self.session_id}")

    # ==================== 消息处理 ====================

    async def handle_audio(self, audio_data: bytes):
        """处理音频数据"""
        # 音频到来 = 可能的打断
        if not self.interrupt_manager.is_interrupted:
            self.interrupt_manager.set_interrupt()

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
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """输入结果回调 - 处理 ASR 或文本输入的结果"""
        text_data: TextData = input_event.event_data

        if not text_data.is_final:
            return

        # 空结果，重置打断标志
        if not text_data.text:
            logger.info(f"ConversationOrchestrator 输入结果为空: session={self.session_id}")
            # 如果输入为空，我们认为这次打断是无效的，不计入上下文拼接
            # 但不重置 interrupt_manager 的 was_interrupted 状态，因为它确实发生过打断动作
            # 只是没有产生有效内容
            return

        # 处理打断场景（仅音频输入有打断逻辑）
        if self.interrupt_manager.was_interrupted:
            # 拼接上一轮和本轮文本
            combined_text = f"{self.turn_context.get('last_user_text', '')} {text_data.text}".strip()
            logger.info(
                f"ConversationOrchestrator 拼接打断文本 (session: {self.session_id}): "
                f"'{combined_text}'"
            )
            user_text = combined_text
        else:
            user_text = text_data.text

        # 更新上下文
        self.turn_context['last_user_text'] = user_text
        # 重置打断记录，为下一轮对话做准备
        self.interrupt_manager.reset_history()

        # 重置当前打断标志，开始新一轮对话生成
        self.interrupt_manager.reset()

        # 触发对话
        await self._trigger_conversation(user_text)

    # ==================== 对话流程 ====================

    async def _trigger_conversation(self, user_text: str):
        """触发对话流程: LLM → TTS"""
        # 从当前会话获取模块
        session_ctx = await self.session_manager.get_session(self.session_id)
        if not session_ctx:
             logger.error(f"ConversationOrchestrator 会话上下文未找到: session={self.session_id}")
             return

        llm_module: BaseLLM = session_ctx.get_module("llm")
        tts_module: BaseTTS = session_ctx.get_module("tts")

        # 模块空值检查
        if not llm_module:
            logger.error(f"ConversationOrchestrator LLM 模块未找到: session={self.session_id}")
            return

        # TTS 模块是可选的，如果未找到则退化为仅文本模式
        if not tts_module:
             logger.info(f"ConversationOrchestrator TTS 模块未启用: session={self.session_id}")

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
        self.sentence_splitter.clear()

        # BaseLLM 的方法是 chat_stream(text, session_id)
        async for text_chunk in llm_module.chat_stream(llm_input, self.session_id):
            content = text_chunk.text

            # 检查打断
            if self.interrupt_manager.is_interrupted:
                logger.info(f"ConversationOrchestrator 对话被打断: session={self.session_id}")
                break

            self.sentence_splitter.append(content)

            # 尝试分割句子
            while True:
                sentence = self.sentence_splitter.split()
                if not sentence:
                    break

                self._create_background_task(
                    self._send_sentence(sentence, tts_module, is_final=False)
                )

        # 发送剩余文本
        remaining = self.sentence_splitter.get_remaining()
        if remaining and not self.interrupt_manager.is_interrupted:
            self._create_background_task(
                self._send_sentence(remaining, tts_module, is_final=True)
            )

    async def _process_text_only(
        self,
        llm_input: TextData,
        llm_module: BaseLLM
    ):
        """只处理 LLM 输出（无 TTS）"""
        async for text_chunk in llm_module.chat_stream(llm_input, self.session_id):
            content = text_chunk.text

            if self.interrupt_manager.is_interrupted:
                break

            if content:
                text_event = StreamEvent(
                    event_type=EventType.SERVER_TEXT_RESPONSE,
                    event_data=TextData(text=content, is_final=False),
                    session_id=self.session_id
                )
                await self.send_callback(text_event)

        # 发送最终标记
        if not self.interrupt_manager.is_interrupted:
            final_event = StreamEvent(
                event_type=EventType.SERVER_TEXT_RESPONSE,
                event_data=TextData(text="", is_final=True),
                session_id=self.session_id
            )
            await self.send_callback(final_event)

    def _create_background_task(self, coro) -> asyncio.Task:
        """创建后台任务并保存引用，防止被垃圾回收

        Args:
            coro: 协程对象

        Returns:
            创建的任务对象
        """
        task = asyncio.create_task(coro)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
        return task

    async def _send_sentence(
        self,
        sentence: str,
        tts_module: BaseTTS,
        is_final: bool = False
    ):
        """发送句子（文本 + 音频）"""
        if self.interrupt_manager.is_interrupted:
            return

        # 发送文本
        text_event = StreamEvent(
            event_type=EventType.SERVER_TEXT_RESPONSE,
            event_data=TextData(text=sentence, is_final=is_final),
            session_id=self.session_id
        )
        await self.send_callback(text_event)

        # BaseTTS.synthesize_stream 返回 AsyncGenerator[AudioData, None]
        # 遍历音频流并发送每个音频块
        async for audio_chunk in tts_module.synthesize_stream(TextData(text=sentence)):
            if self.interrupt_manager.is_interrupted:
                break

            if audio_chunk and audio_chunk.data:
                audio_event = StreamEvent(
                    event_type=EventType.SERVER_AUDIO_RESPONSE,
                    event_data=audio_chunk,
                    session_id=self.session_id
                )
                await self.send_callback(audio_event)
