import asyncio
import time
from collections import deque
from typing import Optional

from core.audio.audio_constants import AudioProcessingConfig


class AudioBufferManager:
    """异步安全的音频缓冲区管理器

    职责:
    - 管理音频数据的缓冲
    - 提供线程安全的异步访问
    - 跟踪最后语音时间
    """

    def __init__(
        self,
        bytes_per_second: int = AudioProcessingConfig.DEFAULT_BYTES_PER_SECOND
    ):
        self._buffer: deque[bytes] = deque()
        self._lock = asyncio.Lock()
        self._bytes_per_second = bytes_per_second
        self._last_speech_time: Optional[float] = None

    async def append(self, chunk: bytes) -> None:
        """添加音频块"""
        async with self._lock:
            self._buffer.append(chunk)
            self._last_speech_time = time.time()

    async def get_buffered_data(self) -> bytes:
        """获取缓冲的音频数据"""
        async with self._lock:
            return b"".join(self._buffer)

    async def clear(self) -> None:
        """清空缓冲区"""
        async with self._lock:
            self._buffer.clear()
            self._last_speech_time = None

    async def get_buffer_duration(self) -> float:
        """获取缓冲区时长（秒）"""
        async with self._lock:
            total_bytes = sum(len(chunk) for chunk in self._buffer)
            return total_bytes / self._bytes_per_second

    async def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        async with self._lock:
            return len(self._buffer) == 0

    async def get_last_speech_time(self) -> Optional[float]:
        """获取最后语音时间"""
        async with self._lock:
            return self._last_speech_time
