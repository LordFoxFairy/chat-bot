"""Protocol (协议) 适配器工厂模块"""

from typing import TYPE_CHECKING, Any, Dict

from backend.core.adapter_registry import AdapterRegistry
from backend.core.interfaces.base_protocol import BaseProtocol

if TYPE_CHECKING:
    from backend.core.session.conversation_manager import ConversationManager

# 创建并配置注册器
protocol_registry: AdapterRegistry[BaseProtocol] = AdapterRegistry("Protocol", BaseProtocol)
protocol_registry.register("websocket", "src.adapters.protocols.websocket_protocol_adapter")


def create_protocol_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
    conversation_manager: "ConversationManager",
) -> BaseProtocol:
    """创建 Protocol 适配器实例

    Protocol 适配器需要额外的 conversation_manager 参数。

    Args:
        adapter_type: 协议类型
        module_id: 模块唯一标识符
        config: 协议配置
        conversation_manager: ConversationManager 实例

    Returns:
        协议适配器实例
    """
    return protocol_registry.create(
        adapter_type=adapter_type,
        module_id=module_id,
        config=config,
        conversation_manager=conversation_manager,
    )
