"""ASR (语音识别) 适配器模块"""

from .asr_factory import asr_registry, create_asr_adapter

__all__ = [
    "asr_registry",
    "create_asr_adapter",
]
