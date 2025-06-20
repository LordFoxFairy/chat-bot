import asyncio
from typing import Type, Optional, Dict, Any
from modules.base_vad import BaseVAD
from core.exceptions import ModuleInitializationError
from utils.logging_setup import logger
from .silero_vad_adapter import SileroVADAdapter

VAD_ADAPTER_REGISTRY: Dict[str, Type[BaseVAD]] = {
    "silero_vad": SileroVADAdapter,
}

def create_vad_adapter(
        module_id: str,
        config: Optional[Dict[str, Any]] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> BaseVAD:
    """
    根据指定的 module_id 创建并返回一个VAD适配器实例。

    参数:
        module_id (str): 要分配给模块实例的ID。
        config (Optional[Dict[str, Any]]): 模块的配置字典。
        event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
        event_manager (Optional[Any]): 事件管理器实例。

    返回:
        BaseVAD: 一个VAD适配器的实例。

    抛出:
        ModuleInitializationError: 如果指定的 adapter_type 不被支持或实例化失败。
    """
    adapter_type = config.get("adapter_type")
    adapter_class = VAD_ADAPTER_REGISTRY.get(adapter_type)
    if not adapter_class:
        raise ModuleInitializationError(
            f"不支持的VAD适配器类型: '{adapter_type}'。 "
            f"可用类型: {list(VAD_ADAPTER_REGISTRY.keys())}"
        )

    try:
        logger.info(
            f"[VAD工厂] 正在为类型 '{adapter_type}' (模块ID: '{module_id}') 创建实例，使用类: '{adapter_class.__name__}'")
        instance = adapter_class(
            module_id=module_id,
            config=config,
            event_loop=event_loop
        )
        return instance
    except Exception as e:
        raise ModuleInitializationError(
            f"创建VAD适配器 '{module_id}' (类型: '{adapter_type}') 失败: {e}"
        ) from e
