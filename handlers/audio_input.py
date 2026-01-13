import asyncio
import re
import time
import uuid
from collections import deque
from typing import Callable, Awaitable, Optional

from modules.base_vad import BaseVAD
from modules.base_asr import BaseASR
from core.session_context import SessionContext
from models import StreamEvent, EventType, AudioData, AudioFormat, TextData
from utils.logging_setup import logger


class AudioInputHandler:
    """音频输入处理器

    职责:
    - 接收音频流
    - VAD 检测和缓冲
    - 触发 ASR 识别
    - 发送识别结果

    特点:
    - 全异步设计
    - 简单直接，无过度抽象
    """

    # 默认配置
    DEFAULT_SILENCE_TIMEOUT = 1.0
    DEFAULT_MAX_BUFFER_DURATION = 5.0
    DEFAULT_MIN_SEGMENT_THRESHOLD = 0.3
    DEFAULT_CHECK_INTERVAL = 0.2
    DEFAULT_BYTES_PER_SECOND = 32000  # 16kHz, 单声道, 16bit
    MAX_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB 最大缓冲区

    # 文本清洗正则
    SPECIAL_TOKENS_PATTERN = re.compile(r'<\|.*?\|>')

    def __init__(
        self,
        session_context: SessionContext,
        result_callback: Callable[[StreamEvent, dict], Awaitable[None]],
        silence_timeout: float = DEFAULT_SILENCE_TIMEOUT,
        max_buffer_duration: float = DEFAULT_MAX_BUFFER_DURATION,
        min_segment_threshold: float = DEFAULT_MIN_SEGMENT_THRESHOLD
    ):
        self.session_context = session_context
        self.result_callback = result_callback

        # 从 session 获取模块（支持会话级隔离）
        self.vad_module = session_context.get_module("vad")
        self.asr_module = session_context.get_module("asr")

        # 配置
        self.silence_timeout = silence_timeout
        self.max_buffer_duration = max_buffer_duration
        self.min_segment_threshold = min_segment_threshold

        # 音频缓冲
        self.audio_buffer: deque[bytes] = deque()
        self.buffer_lock = asyncio.Lock()
        self.last_speech_time: Optional[float] = None

        # 状态管理
        self.is_processing = False
        self.transcript_segments: list[str] = []

        # 任务管理
        self.monitor_task: Optional[asyncio.Task] = None
        self.client_speech_ended = asyncio.Event()

    def start(self) -> None:
        """启动音频处理"""
        if self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"[AudioInput] Started for session {self.session_context.session_id}")

    async def stop(self) -> None:
        """停止音频处理"""
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
            logger.info(f"[AudioInput] Stopped for session {self.session_context.session_id}")

    async def process_chunk(self, chunk: bytes) -> None:
        """处理音频块

        Args:
            chunk: 音频数据
        """
        # VAD 检测
        is_speech = await self.vad_module.detect(chunk)

        if is_speech:
            async with self.buffer_lock:
                # 检查缓冲区大小限制
                current_size = sum(len(c) for c in self.audio_buffer)
                if current_size + len(chunk) > self.MAX_BUFFER_SIZE:
                    logger.warning(
                        f"[AudioInput] Buffer overflow risk, clearing buffer. "
                        f"session={self.session_context.session_id}, size={current_size}"
                    )
                    self.audio_buffer.clear()

                self.audio_buffer.append(chunk)
                self.last_speech_time = time.time()
            logger.debug(f"[AudioInput] Speech detected, session={self.session_context.session_id}")
        else:
            logger.debug(f"[AudioInput] No speech, session={self.session_context.session_id}")

    def signal_client_speech_end(self) -> None:
        """客户端信号：语音结束"""
        logger.info(f"[AudioInput] Client speech end signal, session={self.session_context.session_id}")
        self.client_speech_ended.set()

    async def _monitor_loop(self) -> None:
        """监控循环 - 定期检查是否应该触发 ASR"""
        try:
            while True:
                # 等待检查间隔或客户端结束信号
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(asyncio.sleep(self.DEFAULT_CHECK_INTERVAL)),
                        asyncio.create_task(self.client_speech_ended.wait())
                    ],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # 取消未完成的任务
                for task in pending:
                    task.cancel()

                # 检查是否应该处理
                client_ended = self.client_speech_ended.is_set()
                await self._check_and_process(client_ended)

                # 重置信号
                if client_ended:
                    self.client_speech_ended.clear()

        except asyncio.CancelledError:
            logger.info(f"[AudioInput] Monitor loop cancelled, session={self.session_context.session_id}")
        except Exception as e:
            logger.error(f"[AudioInput] Monitor loop error, session={self.session_context.session_id}: {e}", exc_info=True)

    async def _check_and_process(self, client_ended: bool) -> None:
        """检查是否应该处理音频段"""

        # 防止并发
        if self.is_processing:
            return

        async with self.buffer_lock:
            if not self.audio_buffer:
                return

            # 计算缓冲区时长
            total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
            buffer_duration = total_bytes / self.DEFAULT_BYTES_PER_SECOND

            # 判断是否触发
            should_process = False
            is_final = False
            reason = ""

            if client_ended:
                should_process = True
                is_final = True
                reason = "client_signal"
            elif (
                self.last_speech_time is not None
                and (time.time() - self.last_speech_time >= self.silence_timeout)
                and buffer_duration >= self.min_segment_threshold
            ):
                should_process = True
                is_final = True
                reason = "silence_timeout"
            elif buffer_duration >= self.max_buffer_duration:
                should_process = True
                is_final = False
                reason = "max_buffer"

            if not should_process:
                return

            logger.info(f"[AudioInput] Processing segment (reason={reason}, is_final={is_final}), session={self.session_context.session_id}")

            # 获取音频数据并清空缓冲
            audio_bytes = b"".join(self.audio_buffer)
            self.audio_buffer.clear()
            self.last_speech_time = None

        # 处理
        try:
            self.is_processing = True
            await self._process_audio_segment(audio_bytes, is_final)
        finally:
            self.is_processing = False

    async def _process_audio_segment(self, audio_bytes: bytes, is_final: bool) -> None:
        """处理音频段 - ASR 识别"""

        if not audio_bytes:
            if is_final:
                await self._send_final_result()
            return

        try:
            # ASR 识别
            audio_data = AudioData(data=audio_bytes, format=AudioFormat.PCM)
            recognized_text = await self.asr_module.recognize(audio_data)

            # 清洗文本
            cleaned_text = self._clean_text(recognized_text)

            if cleaned_text:
                logger.info(f"[AudioInput] ASR result: '{cleaned_text}', session={self.session_context.session_id}")
                self.transcript_segments.append(cleaned_text)

            # 最终段发送结果
            if is_final:
                await self._send_final_result()

        except Exception as e:
            logger.error(f"[AudioInput] ASR processing failed, session={self.session_context.session_id}: {e}", exc_info=True)
            if is_final:
                await self._send_final_result()

    async def _send_final_result(self) -> None:
        """发送最终识别结果"""

        final_text = " ".join(seg for seg in self.transcript_segments if seg).strip()

        logger.info(f"[AudioInput] Final transcript: '{final_text}', session={self.session_context.session_id}")

        event = StreamEvent(
            event_type=EventType.ASR_RESULT,
            event_data=TextData(
                text=final_text,
                is_final=True,
                message_id=str(uuid.uuid4())
            ),
            session_id=self.session_context.session_id,
            tag_id=self.session_context.tag_id
        )

        await self.result_callback(event, {"session_id": self.session_context.session_id})

        # 清空
        self.transcript_segments.clear()

    def _clean_text(self, text: str) -> str:
        """清洗文本 - 移除特殊标记"""
        if not text:
            return ""
        cleaned = self.SPECIAL_TOKENS_PATTERN.sub('', text).strip()
        return cleaned
