"""TTS (文本转语音) 适配器模块"""

from .tts_factory import tts_registry, create_tts_adapter

__all__ = [
    "tts_registry",
    "create_tts_adapter",
]
