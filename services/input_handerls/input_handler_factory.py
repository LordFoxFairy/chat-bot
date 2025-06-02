import logging  # 导入 logging 模块
from typing import Type, Optional, Dict, Any

# 核心框架组件的导入
# from core_framework.session_manager import SessionManager # WebsocketInputHandler 不再直接依赖
# from core_framework.event_manager import EventManager # WebsocketInputHandler 不再直接依赖
from core.chat_engine import ChatEngine  # WebsocketInputHandler 现在依赖 ChatEngine
from core.exceptions import ConfigurationError

# 输入处理器基类和具体实现的导入
from .base_input_handler import BaseInputHandler
from .websocket_input_handler import WebsocketInputHandler

# 如果有其他输入处理器，例如 MqttInputHandler，也在这里导入
# from .mqtt_input_handler import MqttInputHandler # 示例

# 日志记录器
logger = logging.getLogger(__name__)

# 输入处理器注册表
INPUT_HANDLER_REGISTRY: Dict[str, Type[BaseInputHandler]] = {
    "websocket": WebsocketInputHandler,
    # "mqtt": MqttInputHandler, # 示例：如果将来添加MQTT处理器
}


def create_input_handler(
        handler_type: str,
        chat_engine: ChatEngine,  # 修改：传递 ChatEngine 实例
        config: Optional[Dict[str, Any]] = None,
        # session_manager: Optional[SessionManager] = None, # 移除，除非其他类型的 handler 需要
        # event_manager: Optional[EventManager] = None,   # 移除，除非其他类型的 handler 需要
        **kwargs: Any  # 为将来可能需要的其他参数保留扩展性
) -> BaseInputHandler:
    """
    根据指定的 handler_type 创建并返回一个输入处理器实例。

    :param handler_type: 要创建的输入处理器的类型字符串 (例如 "websocket")。
    :param chat_engine: ChatEngine 的实例，WebsocketInputHandler 需要它。
    :param config: (可选) 特定于该输入处理器的配置字典。
    :param kwargs: (可选) 传递给特定输入处理器构造函数的其他关键字参数。
    :return: BaseInputHandler 的实例。
    :raises ConfigurationError: 如果处理器类型不支持或创建失败。
    """
    handler_class = INPUT_HANDLER_REGISTRY.get(handler_type)
    if not handler_class:
        error_msg = (
            f"不支持的输入处理器类型: '{handler_type}'. "
            f"可用类型: {list(INPUT_HANDLER_REGISTRY.keys())}"
        )
        logger.error(f"[InputHandlerFactory] {error_msg}")
        raise ConfigurationError(error_msg)

    try:
        logger.info(f"[InputHandlerFactory] 正在为类型 '{handler_type}' 创建实例，使用类: '{handler_class.__name__}'")

        # 构造参数字典
        # WebsocketInputHandler 的构造函数现在是 __init__(self, chat_engine: ChatEngine)
        # 它不再直接接收 config, session_manager, event_manager
        # 如果其他类型的 handler 有不同的构造函数签名，需要在这里处理

        instance_kwargs: Dict[str, Any] = {}

        if handler_type == "websocket":
            # WebsocketInputHandler 特定的构造
            instance_kwargs['chat_engine'] = chat_engine
            # 如果 WebsocketInputHandler 将来需要自己的配置，可以在这里传递
            # if config:
            #     instance_kwargs['config'] = config
        else:
            # 对于其他类型的 handler，它们可能仍然需要 config, session_manager, event_manager
            # 或者其他的参数。这里提供一个通用模式，但需要根据具体 handler 调整。
            # instance_kwargs['session_manager'] = session_manager
            # instance_kwargs['event_manager'] = event_manager
            # instance_kwargs['config'] = config if config is not None else {}
            # instance_kwargs.update(kwargs) # 合并额外的kwargs
            logger.warning(f"[InputHandlerFactory] 类型 '{handler_type}' 的处理器构造逻辑可能需要调整以适应其参数。")
            # 默认传递 chat_engine 和 config，其他类型的 handler 如果需要不同参数，需要在这里扩展逻辑
            instance_kwargs['chat_engine'] = chat_engine  # 假设其他handler也可能需要
            instance_kwargs['config'] = config if config is not None else {}

        instance = handler_class(**instance_kwargs)  # type: ignore

        logger.info(f"[InputHandlerFactory] 成功创建输入处理器实例 (类型: '{handler_type}')。")
        return instance
    except Exception as e:
        error_msg = f"创建输入处理器 (类型: '{handler_type}') 失败: {e}"
        logger.exception(f"[InputHandlerFactory] {error_msg}")  # 使用 logger.exception 记录堆栈信息
        raise ConfigurationError(error_msg) from e

