from typing import Dict, Type, Any

from core.exceptions import ModuleInitializationError
from modules.base_llm import BaseLLM
from utils.logging_setup import logger

from .langchain_llm_adapter import LangChainLLMAdapter


# LLM 适配器注册表
LLM_ADAPTERS: Dict[str, Type[BaseLLM]] = {
    "langchain": LangChainLLMAdapter,
    # 未来可以添加更多 LLM 适配器:
    # "openai_direct": OpenAIDirectAdapter,
}


def create_llm_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
) -> BaseLLM:
    """创建 LLM 适配器实例"""
    adapter_class = LLM_ADAPTERS.get(adapter_type)

    if adapter_class is None:
        available_types = list(LLM_ADAPTERS.keys())
        raise ModuleInitializationError(
            f"不支持的 LLM 适配器类型: '{adapter_type}'. "
            f"可用类型: {available_types}"
        )

    try:
        logger.info(
            f"LLM Factory: 创建 '{adapter_type}' 适配器，"
            f"模块ID: {module_id}，类: {adapter_class.__name__}"
        )

        instance = adapter_class(
            module_id=module_id,
            config=config,
        )

        return instance

    except Exception as e:
        raise ModuleInitializationError(
            f"创建 LLM 适配器 '{adapter_type}' 失败: {e}"
        ) from e
