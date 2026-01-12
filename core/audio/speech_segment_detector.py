import time
from dataclasses import dataclass
from typing import Optional

from core.audio.audio_constants import AudioProcessingConfig
from data_models import SegmentDetectionResult


@dataclass
class SegmentDetectionConfig:
    """段检测配置"""

    silence_timeout: float = AudioProcessingConfig.DEFAULT_SILENCE_TIMEOUT
    max_buffer_duration: float = AudioProcessingConfig.DEFAULT_MAX_BUFFER_DURATION
    min_segment_threshold: float = AudioProcessingConfig.DEFAULT_MIN_SEGMENT_THRESHOLD


class SpeechSegmentDetector:
    """语音段检测器

    职责:
    - 判断何时应该触发 ASR
    - 基于静音超时、最大缓冲等策略
    """

    def __init__(self, config: SegmentDetectionConfig):
        self._config = config

    async def should_process(
        self,
        buffer_duration: float,
        last_speech_time: Optional[float],
        client_ended: bool
    ) -> SegmentDetectionResult:
        """判断是否应该处理当前音频段

        Args:
            buffer_duration: 当前缓冲区时长（秒）
            last_speech_time: 最后一次语音时间戳
            client_ended: 客户端是否发送了结束信号

        Returns:
            SegmentDetectionResult: 检测结果
        """

        # 场景1: 客户端明确发送结束信号
        if client_ended:
            return SegmentDetectionResult(
                should_process=True,
                is_final=True,
                reason="client_signal"
            )

        # 场景2: 后端 VAD 检测到静音超时
        if (
            last_speech_time is not None
            and (time.time() - last_speech_time >= self._config.silence_timeout)
            and buffer_duration >= self._config.min_segment_threshold
        ):
            return SegmentDetectionResult(
                should_process=True,
                is_final=True,
                reason="silence_timeout"
            )

        # 场景3: 缓冲区达到最大时长（中间段）
        if buffer_duration >= self._config.max_buffer_duration:
            return SegmentDetectionResult(
                should_process=True,
                is_final=False,
                reason="max_buffer"
            )

        # 场景4: 不需要处理
        return SegmentDetectionResult(
            should_process=False,
            is_final=False,
            reason="waiting"
        )
