"""LLM (大语言模型) 适配器模块"""

from .llm_factory import llm_registry, create_llm_adapter

__all__ = [
    "llm_registry",
    "create_llm_adapter",
]
