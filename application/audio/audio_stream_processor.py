import asyncio
from typing import Callable, Awaitable, Optional

from modules.base_vad import BaseVAD
from modules.base_asr import BaseASR
from core.audio.audio_buffer_manager import AudioBufferManager
from core.audio.speech_segment_detector import SpeechSegmentDetector, SegmentDetectionConfig
from core.audio.audio_constants import AudioProcessingConfig
from core.text.text_post_processor import TextPostProcessor
from core.session_context import SessionContext
from application.audio.audio_processing_orchestrator import AudioProcessingOrchestrator
from data_models import StreamEvent
from utils.logging_setup import logger


class AudioStreamProcessor:
    """音频流处理器（重构后的 AudioConsumer）

    职责:
    - 管理异步监控任务的生命周期
    - 协调各组件工作
    - 提供简单的外部接口

    特点:
    - 全异步设计
    - 职责单一
    - 使用 asyncio.Lock
    - 组件化架构
    """

    def __init__(
        self,
        session_context: SessionContext,
        vad_module: BaseVAD,
        asr_module: BaseASR,
        result_callback: Callable[[StreamEvent, dict], Awaitable[None]],
        silence_timeout: float = AudioProcessingConfig.DEFAULT_SILENCE_TIMEOUT,
        max_buffer_duration: float = AudioProcessingConfig.DEFAULT_MAX_BUFFER_DURATION,
        min_segment_threshold: float = AudioProcessingConfig.DEFAULT_MIN_SEGMENT_THRESHOLD
    ):
        self._session = session_context

        # 配置
        self._config = SegmentDetectionConfig(
            silence_timeout=silence_timeout,
            max_buffer_duration=max_buffer_duration,
            min_segment_threshold=min_segment_threshold
        )

        # 组件初始化
        self._buffer = AudioBufferManager()
        self._detector = SpeechSegmentDetector(self._config)
        self._text_processor = TextPostProcessor()

        self._orchestrator = AudioProcessingOrchestrator(
            session_context=session_context,
            vad_module=vad_module,
            asr_module=asr_module,
            buffer_manager=self._buffer,
            segment_detector=self._detector,
            text_processor=self._text_processor,
            result_callback=result_callback
        )

        # 监控任务
        self._monitor_task: Optional[asyncio.Task] = None
        self._client_speech_ended = asyncio.Event()

    def start(self) -> None:
        """启动音频流处理"""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info(
                f"[AudioStreamProcessor] Started for session {self._session.session_id}"
            )

    def stop(self) -> None:
        """停止音频流处理"""
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
            logger.info(
                f"[AudioStreamProcessor] Stopped for session {self._session.session_id}"
            )

    async def process_chunk(self, chunk: bytes) -> None:
        """处理音频块（外部调用）

        Args:
            chunk: 音频块数据
        """
        await self._orchestrator.process_chunk_with_vad(chunk)

    def signal_client_speech_end(self) -> None:
        """客户端信号：语音结束"""
        logger.info(
            f"[AudioStreamProcessor] Client speech end signal "
            f"for session {self._session.session_id}"
        )
        self._client_speech_ended.set()

    async def _monitor_loop(self) -> None:
        """监控循环 - 定期检查是否应该处理音频段"""
        try:
            while True:
                # 等待检查间隔或客户端结束信号
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(
                            asyncio.sleep(AudioProcessingConfig.DEFAULT_CHECK_INTERVAL)
                        ),
                        asyncio.create_task(self._client_speech_ended.wait())
                    ],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # 取消未完成的任务
                for task in pending:
                    task.cancel()

                # 检查是否应该处理
                client_ended = self._client_speech_ended.is_set()
                await self._orchestrator.check_and_process_segment(
                    client_ended=client_ended
                )

                # 重置客户端结束信号
                if client_ended:
                    self._client_speech_ended.clear()

        except asyncio.CancelledError:
            logger.info(
                f"[AudioStreamProcessor] Monitor loop cancelled "
                f"for session {self._session.session_id}"
            )
        except Exception as e:
            logger.error(
                f"[AudioStreamProcessor] Monitor loop error "
                f"for session {self._session.session_id}: {e}",
                exc_info=True
            )
