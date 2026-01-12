from dataclasses import dataclass


@dataclass(frozen=True)
class AudioProcessingConfig:
    """音频处理配置常量"""

    # 检查间隔 (秒)
    DEFAULT_CHECK_INTERVAL: float = 0.2

    # 静音超时 (秒)
    DEFAULT_SILENCE_TIMEOUT: float = 1.0

    # 最大缓冲时长 (秒)
    DEFAULT_MAX_BUFFER_DURATION: float = 5.0

    # 最小音频段阈值 (秒)
    DEFAULT_MIN_SEGMENT_THRESHOLD: float = 0.3

    # 采样率：16kHz, 单声道, 16bit = 32000 bytes/sec
    DEFAULT_BYTES_PER_SECOND: int = 32000


@dataclass(frozen=True)
class TextCleaningPatterns:
    """文本清洗规则"""

    # FunASR SenseVoice 特殊标记
    SPECIAL_TOKENS_PATTERN: str = r'<\|.*?\|>'
