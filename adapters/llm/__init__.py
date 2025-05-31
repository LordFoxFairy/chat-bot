# adapters/llm/__init__.py
from .llm_factory import create_llm_adapter, LLM_ADAPTER_REGISTRY

__all__ = [
    "create_llm_adapter",
    "LLM_ADAPTER_REGISTRY",
]