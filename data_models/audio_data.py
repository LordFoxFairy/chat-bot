from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import time # 确保 time 被导入

class AudioFormat(Enum):
    """
    音频格式枚举。
    """
    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    OPUS = "opus" # 通常用于 WebRTC 和流式传输



class AudioData(BaseModel):
    """
    音频数据模型。
    """
    data: bytes = Field(..., description="原始音频数据。")
    format: AudioFormat = Field(..., description="音频数据的格式。")
    channels: int = Field(..., description="音频通道数。")
    sample_rate: int = Field(..., description="采样率 (Hz)。")
    sample_width: int = Field(..., description="采样位深/样本宽度 (字节，例如 16位音频为2)。")
    # source: AudioSource = Field(AudioSource.STREAM, description="音频来源。") # 根据用户要求移除
    timestamp: Optional[float] = Field(default_factory=time.time, description="音频数据生成的时间戳 (纪元秒)。")
    chunk_id: Optional[str] = Field(None, description="音频块的唯一标识符，用于流式传输。")
    is_final: bool = Field(False, description="指示这是否是流的最后一个音频块。")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="任何附加的元数据。")

    def __str__(self):
        """
        返回 AudioData 对象的字符串表示形式。
        """
        return (f"AudioData(格式={self.format.value}, 通道数={self.channels}, "
                f"采样率={self.sample_rate}, 采样位深={self.sample_width}, "
                # f"来源='{self.source.value}', " # 根据用户要求移除
                f"数据长度={len(self.data)}, "
                f"时间戳={self.timestamp}, 块ID={self.chunk_id}, 是否最后一块={self.is_final_chunk})")

