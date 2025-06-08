from typing import Type, Optional, Dict, Any

# 输入处理器基类和具体实现的导入
from adapters.handlers.websocket_handler import WebSocketServerHandler
# 核心框架组件的导入
from core.exceptions import ConfigurationError
from modules.base_handler import BaseHandler
from utils.logging_setup import logger

# 输入处理器注册表
INPUT_HANDLER_REGISTRY: Dict[str, Type[BaseHandler]] = {
    "websocket": WebSocketServerHandler,
}


def create_input_handler(
        module_id: str,
        config: Optional[Dict[str, Any]] = None
) -> BaseHandler:
    """
    根据指定的 handler_type 创建并返回一个输入处理器实例。

        module_id (str): 要分配给模块实例的 ID。
        config (Optional[Dict[str, Any]]): 模块的配置字典。
        event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
    """
    handler_type = config.get("adapter_type")
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

        instance = handler_class(
            module_id=module_id,
            config=config
        )

        logger.info(f"[InputHandlerFactory] 成功创建输入处理器实例 (类型: '{handler_type}')。")
        return instance
    except Exception as e:
        error_msg = f"创建输入处理器 (类型: '{handler_type}') 失败: {e}"
        logger.exception(f"[InputHandlerFactory] {error_msg}")  # 使用 logger.exception 记录堆栈信息
        raise ConfigurationError(error_msg) from e
