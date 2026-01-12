import uuid
from typing import Callable, Awaitable, List

from modules.base_vad import BaseVAD
from modules.base_asr import BaseASR
from core.audio.audio_buffer_manager import AudioBufferManager
from core.audio.speech_segment_detector import SpeechSegmentDetector
from core.text.text_post_processor import TextPostProcessor
from core.session_context import SessionContext
from data_models import StreamEvent, EventType, AudioData, AudioFormat, TextData
from utils.logging_setup import logger


class AudioProcessingOrchestrator:
    """音频处理编排器

    职责:
    - 协调 VAD、ASR、缓冲区之间的交互
    - 管理音频处理流程
    - 发送 ASR 结果事件
    """

    def __init__(
        self,
        session_context: SessionContext,
        vad_module: BaseVAD,
        asr_module: BaseASR,
        buffer_manager: AudioBufferManager,
        segment_detector: SpeechSegmentDetector,
        text_processor: TextPostProcessor,
        result_callback: Callable[[StreamEvent, dict], Awaitable[None]]
    ):
        self._session = session_context
        self._vad = vad_module
        self._asr = asr_module
        self._buffer = buffer_manager
        self._detector = segment_detector
        self._text_processor = text_processor
        self._result_callback = result_callback

        self._transcript_segments: List[str] = []
        self._is_processing = False

    async def process_chunk_with_vad(self, chunk: bytes) -> None:
        """处理音频块（带 VAD 过滤）

        Args:
            chunk: 音频块数据
        """
        # VAD 检测
        is_speech = await self._vad.detect(chunk)

        if is_speech:
            await self._buffer.append(chunk)
            logger.debug(
                f"[Orchestrator] Session {self._session.session_id}: "
                f"Speech detected, chunk appended"
            )
        else:
            logger.debug(
                f"[Orchestrator] Session {self._session.session_id}: "
                f"No speech detected, chunk discarded"
            )

    async def check_and_process_segment(
        self,
        client_ended: bool = False
    ) -> None:
        """检查是否应该处理音频段

        Args:
            client_ended: 客户端是否发送了结束信号
        """

        # 防止并发处理
        if self._is_processing:
            return

        # 检查缓冲区
        if await self._buffer.is_empty():
            return

        # 获取检测参数
        buffer_duration = await self._buffer.get_buffer_duration()
        last_speech_time = await self._buffer.get_last_speech_time()

        # 判断是否应该处理
        result = await self._detector.should_process(
            buffer_duration=buffer_duration,
            last_speech_time=last_speech_time,
            client_ended=client_ended
        )

        if not result.should_process:
            return

        logger.info(
            f"[Orchestrator] Session {self._session.session_id}: "
            f"Processing segment (reason: {result.reason}, is_final: {result.is_final})"
        )

        # 处理音频段
        try:
            self._is_processing = True
            await self._process_audio_segment(result.is_final)
        finally:
            self._is_processing = False

    async def _process_audio_segment(self, is_final: bool) -> None:
        """处理音频段 - ASR 识别

        Args:
            is_final: 是否为最终段
        """

        # 获取并清空缓冲区
        audio_bytes = await self._buffer.get_buffered_data()
        await self._buffer.clear()

        if not audio_bytes:
            logger.warning(
                f"[Orchestrator] Session {self._session.session_id}: "
                f"Audio buffer is empty"
            )
            # 如果是最终段，即使为空也要发送回调
            if is_final:
                await self._send_final_result()
            return

        try:
            # 调用 ASR
            audio_data = AudioData(data=audio_bytes, format=AudioFormat.PCM)
            recognized_text = await self._asr.recognize(audio_data)

            # 清洗文本
            cleaned_text = self._text_processor.clean(recognized_text)

            if cleaned_text:
                logger.info(
                    f"[Orchestrator] Session {self._session.session_id}: "
                    f"ASR result: '{cleaned_text}'"
                )
                self._transcript_segments.append(cleaned_text)

            # 如果是最终段，发送完整转录
            if is_final:
                await self._send_final_result()

        except Exception as e:
            logger.error(
                f"[Orchestrator] Session {self._session.session_id}: "
                f"ASR processing failed: {e}",
                exc_info=True
            )
            # 即使出错，如果是最终段也要发送结果
            if is_final:
                await self._send_final_result()

    async def _send_final_result(self) -> None:
        """发送最终转录结果"""

        final_text = self._text_processor.merge_segments(self._transcript_segments)

        logger.info(
            f"[Orchestrator] Session {self._session.session_id}: "
            f"Final transcript: '{final_text}'"
        )

        event = StreamEvent(
            event_type=EventType.ASR_RESULT,
            event_data=TextData(
                text=final_text,
                is_final=True,
                message_id=str(uuid.uuid4())
            ),
            session_id=self._session.session_id,
            tag_id=self._session.tag_id
        )

        await self._result_callback(event, {"session_id": self._session.session_id})

        # 清空转录段
        self._transcript_segments.clear()
