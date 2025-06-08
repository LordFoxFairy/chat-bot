import asyncio
from typing import Type, Optional, Dict, Any
from modules.base_llm import BaseLLM
from core.exceptions import ModuleInitializationError
from utils.logging_setup import logger
from .langchain_llm_adapter import LangchainLLMAdapter

LLM_ADAPTER_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "langchain": LangchainLLMAdapter,
}


def create_llm_adapter(
        module_id: str,
        config: Optional[Dict[str, Any]] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> BaseLLM:
    """
    根据指定的 adapter_type 创建并返回一个 LLM 适配器实例。

    参数:
        module_id (str): 要分配给模块实例的 ID。
        config (Optional[Dict[str, Any]]): 模块的配置字典。
        event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环

    返回:
        BaseLLM: 一个 LLM 适配器的实例。

    抛出:
        ModuleInitializationError: 如果指定的 adapter_type 不被支持或实例化失败。
    """
    adapter_type = config.get("adapter_type")
    adapter_class = LLM_ADAPTER_REGISTRY.get(adapter_type)
    if not adapter_class:
        raise ModuleInitializationError(
            f"不支持的 LLM 适配器类型: '{adapter_type}'. "
            f"可用类型: {list(LLM_ADAPTER_REGISTRY.keys())}"
        )

    try:
        logger.info(
            f"[LLM Factory] 正在为类型 '{adapter_type}' 创建实例 '{module_id}' 使用类 '{adapter_class.__name__}'")
        # BaseLLM 和 LangchainLLMAdapter 的构造函数都需要 session_manager
        instance = adapter_class(
            module_id=module_id,
            config=config,
            event_loop=event_loop,
        )
        return instance
    except Exception as e:
        raise ModuleInitializationError(
            f"创建 LLM 适配器 '{module_id}' (类型: '{adapter_type}') 失败: {e}"
        ) from e
