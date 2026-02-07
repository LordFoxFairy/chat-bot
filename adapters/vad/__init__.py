"""VAD (语音活动检测) 适配器模块"""

from .vad_factory import vad_registry, create_vad_adapter

__all__ = [
    "vad_registry",
    "create_vad_adapter",
]
