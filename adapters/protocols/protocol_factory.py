from typing import Dict, Type, Any

from core.exceptions import ModuleInitializationError
from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger

from .websocket_protocol_adapter import WebSocketProtocolAdapter


# Protocol 适配器注册表
PROTOCOL_ADAPTERS: Dict[str, Type[BaseProtocol]] = {
    "websocket": WebSocketProtocolAdapter,
    # 未来可以添加更多协议适配器:
    # "http": HTTPProtocolAdapter,
    # "grpc": GRPCProtocolAdapter,
}


def create_protocol_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
) -> BaseProtocol:
    """创建 Protocol 适配器实例"""
    adapter_class = PROTOCOL_ADAPTERS.get(adapter_type)

    if adapter_class is None:
        available_types = list(PROTOCOL_ADAPTERS.keys())
        raise ModuleInitializationError(
            f"不支持的 Protocol 适配器类型: '{adapter_type}'. "
            f"可用类型: {available_types}"
        )

    try:
        logger.info(
            f"Protocol Factory: 创建 '{adapter_type}' 适配器，"
            f"模块ID: {module_id}，类: {adapter_class.__name__}"
        )

        instance = adapter_class(
            module_id=module_id,
            config=config,
        )

        return instance

    except Exception as e:
        raise ModuleInitializationError(
            f"创建 Protocol 适配器 '{adapter_type}' 失败: {e}"
        ) from e
