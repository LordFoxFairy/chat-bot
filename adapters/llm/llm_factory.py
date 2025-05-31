import asyncio
from typing import Type, Optional, Dict, Any

from modules.base_llm import BaseLLM #
from core_framework.exceptions import ModuleInitializationError #
# LLM 适配器
from .langchain_llm_adapter import LangchainLLMAdapter #


# 如果需要更正式的 EventManager 和 SessionManager 类型提示:
# from core_framework.event_manager import EventManager
# from core_framework.session_manager import SessionManager

LLM_ADAPTER_REGISTRY: Dict[str, Type[BaseLLM]] = {
    "langchain": LangchainLLMAdapter,
}

def create_llm_adapter(
    module_id: str,
    config: Optional[Dict[str, Any]] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    event_manager: Optional[Any] = None, # 使用 Any 避免循环导入 EventManager
    session_manager: Optional[Any] = None # 使用 Any 避免循环导入 SessionManager
) -> BaseLLM:
    """
    根据指定的 adapter_type 创建并返回一个 LLM 适配器实例。

    参数:
        module_id (str): 要分配给模块实例的 ID。
        config (Optional[Dict[str, Any]]): 模块的配置字典。
        event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
        event_manager (Optional[Any]): 事件管理器实例。
        session_manager (Optional[Any]): 会话管理器实例，对LLM至关重要。

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
        print(f"[LLM Factory] 正在为类型 '{adapter_type}' 创建实例 '{module_id}' 使用类 '{adapter_class.__name__}'")
        # BaseLLM 和 LangchainLLMAdapter 的构造函数都需要 session_manager
        instance = adapter_class(
            module_id=module_id,
            config=config,
            event_loop=event_loop,
            event_manager=event_manager,
            session_manager=session_manager # 传递 session_manager
        )
        return instance
    except Exception as e:
        raise ModuleInitializationError(
            f"创建 LLM 适配器 '{module_id}' (类型: '{adapter_type}') 失败: {e}"
        ) from e