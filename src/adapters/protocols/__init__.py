"""Protocol (协议) 适配器模块"""

from .protocol_factory import protocol_registry, create_protocol_adapter

__all__ = [
    "protocol_registry",
    "create_protocol_adapter",
]
