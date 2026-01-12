from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSegment:
    """音频段值对象"""

    data: bytes
    duration: float
    is_final: bool
    timestamp: float


@dataclass(frozen=True)
class SegmentDetectionResult:
    """段检测结果"""

    should_process: bool
    is_final: bool
    reason: str  # "silence_timeout" | "max_buffer" | "client_signal" | "waiting"
