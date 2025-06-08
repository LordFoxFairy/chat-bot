from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


class AudioFormat(str,Enum):
    """
    音频格式枚举。
    """
    PCM = "pcm"
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    OPUS = "opus"  # 通常用于 WebRTC 和流式传输


class AudioData(BaseModel):
    """
    音频数据模型。
    """
    message_id: Optional[str] = Field(None, description="消息id")
    data: bytes = Field(..., description="原始音频数据。")
    format: AudioFormat = Field(..., description="音频数据的格式。")
    # 注意：通常 channels 默认为 1 (单声道) 或 2 (立体声)
    channels: Optional[int] = Field(2, description="音频声道数")
    sample_rate: Optional[int] = Field(16000, description="音频采样率")
    sample_width: Optional[int] = Field(2, description="音频采样宽度（字节数）")
    is_final: bool = Field(False, description="指示这是否是流的最后一个音频块。")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="任何附加的元数据。")

    def __str__(self):
        """
        返回 AudioData 对象的字符串表示形式。
        """
        return (f"AudioData(格式={self.format.value}, "
                f"数据长度={len(self.data)}, "
                f"时间戳={self.timestamp}, 是否最后一块={self.is_final_chunk})")