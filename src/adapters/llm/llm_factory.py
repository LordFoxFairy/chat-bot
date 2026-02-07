"""LLM (大语言模型) 适配器工厂模块"""

from typing import Any, Dict

from src.core.adapter_registry import AdapterRegistry, create_factory_function
from src.core.interfaces.base_llm import BaseLLM

# 创建 LLM 适配器注册器
llm_registry: AdapterRegistry[BaseLLM] = AdapterRegistry("LLM", BaseLLM)

# 注册可用的 LLM 适配器
llm_registry.register(
    "langchain",
    "src.adapters.llm.langchain_llm_adapter"
)
llm_registry.register(
    "langchain_agent",
    "src.adapters.llm.langchain_agent_adapter"
)

# 向后兼容：导出工厂函数
create_llm_adapter = create_factory_function(llm_registry)

# 向后兼容：导出适配器加载器字典
LLM_ADAPTER_LOADERS = {
    adapter_type: lambda at=adapter_type: llm_registry._loaders[at]()
    for adapter_type in llm_registry.available_types
}
