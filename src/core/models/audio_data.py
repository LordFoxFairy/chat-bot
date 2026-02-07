"""音频数据模型"""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class AudioFormat(str, Enum):
    """音频格式枚举"""

    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    OPUS = "opus"


# 支持的采样率
VALID_SAMPLE_RATES = frozenset({8000, 16000, 24000, 44100, 48000})


class AudioData(BaseModel):
    """音频数据模型

    Attributes:
        data: 原始音频数据
        format: 音频格式
        channels: 声道数 (1=单声道, 2=立体声)
        sample_rate: 采样率 (Hz)
        sample_width: 采样宽度 (字节，2=16bit)
        is_final: 是否为最后一个音频块
    """
    model_config = {"frozen": True}

    message_id: Optional[str] = None
    data: bytes = Field(..., min_length=1, description="原始音频数据")
    format: AudioFormat = Field(..., description="音频格式")
    channels: int = Field(1, ge=1, le=8, description="声道数")
    sample_rate: int = Field(16000, description="采样率 (Hz)")
    sample_width: int = Field(2, description="采样宽度 (字节)")
    is_final: bool = Field(False, description="是否为最后一个音频块")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        if v not in VALID_SAMPLE_RATES:
            raise ValueError(f"采样率必须是 {sorted(VALID_SAMPLE_RATES)} 之一")
        return v

    @field_validator("sample_width")
    @classmethod
    def validate_sample_width(cls, v: int) -> int:
        if v not in {1, 2, 4}:
            raise ValueError("采样宽度必须是 1, 2 或 4 字节")
        return v

    @property
    def size_bytes(self) -> int:
        """返回音频数据大小（字节）"""
        return len(self.data)

    @property
    def duration_seconds(self) -> float:
        """返回音频时长（秒）"""
        bytes_per_second = self.sample_rate * self.channels * self.sample_width
        if bytes_per_second == 0:
            return 0.0
        return len(self.data) / bytes_per_second

    def __str__(self) -> str:
        return (
            f"AudioData(format={self.format.value}, "
            f"size={len(self.data)}, "
            f"rate={self.sample_rate}, "
            f"final={self.is_final})"
        )
