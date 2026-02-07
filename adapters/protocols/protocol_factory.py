"""Protocol (协议) 适配器工厂模块"""

from typing import TYPE_CHECKING, Any, Dict

from core.adapter_registry import AdapterRegistry
from core.exceptions import ModuleInitializationError
from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger

if TYPE_CHECKING:
    from core.conversation_manager import ConversationManager

# 创建 Protocol 适配器注册器
protocol_registry: AdapterRegistry[BaseProtocol] = AdapterRegistry("Protocol", BaseProtocol)

# 注册可用的 Protocol 适配器
protocol_registry.register(
    "websocket",
    "adapters.protocols.websocket_protocol_adapter"
)
# 未来可以添加更多协议适配器:
# protocol_registry.register("http", "adapters.protocols.http_protocol_adapter")
# protocol_registry.register("grpc", "adapters.protocols.grpc_protocol_adapter")


def create_protocol_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
    conversation_manager: "ConversationManager"
) -> BaseProtocol:
    """创建 Protocol 适配器实例

    Protocol 适配器需要额外的 conversation_manager 参数，
    因此不使用通用的 create_factory_function。

    Args:
        adapter_type: 协议类型 (websocket, http, grpc 等)
        module_id: 模块唯一标识符
        config: 协议配置
        conversation_manager: ConversationManager 实例，用于会话管理

    Returns:
        BaseProtocol: 协议适配器实例

    Raises:
        ModuleInitializationError: 当适配器类型不支持或创建失败时
    """
    return protocol_registry.create(
        adapter_type=adapter_type,
        module_id=module_id,
        config=config,
        conversation_manager=conversation_manager
    )


# 向后兼容：导出适配器加载器字典
PROTOCOL_ADAPTER_LOADERS = {
    adapter_type: lambda at=adapter_type: protocol_registry._loaders[at]()
    for adapter_type in protocol_registry.available_types
}
