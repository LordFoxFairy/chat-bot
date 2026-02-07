"""LLM (大语言模型) 适配器工厂模块"""

from src.core.adapter_registry import AdapterRegistry, create_factory_function
from src.core.interfaces.base_llm import BaseLLM

# 创建并配置注册器
llm_registry: AdapterRegistry[BaseLLM] = AdapterRegistry("LLM", BaseLLM)
llm_registry.register("langchain", "src.adapters.llm.langchain_llm_adapter")

# 导出工厂函数
create_llm_adapter = create_factory_function(llm_registry)
