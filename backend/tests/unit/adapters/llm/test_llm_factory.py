import pytest
import asyncio
from unittest.mock import MagicMock, patch

from backend.adapters.llm.llm_factory import create_llm_adapter
from backend.core.interfaces.base_llm import BaseLLM
from backend.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter

class TestLLMFactory:

    def test_create_llm_adapter_success(self):
        """测试创建工厂函数"""
        config = {
            "model_name": "gpt-4",
            "api_key": "test-key"
        }

        # 验证 registry 注册
        adapter = create_llm_adapter("langchain", "llm-mod", config)

        assert isinstance(adapter, LangChainLLMAdapter)
        assert adapter.module_id == "llm-mod"
        assert adapter.model_name == "gpt-4"

    def test_create_unknown_adapter(self):
        """测试创建未知适配器"""
        from backend.core.models.exceptions import ModuleInitializationError
        # 修正: 工厂抛出 ModuleInitializationError 而不是 ValueError
        with pytest.raises(ModuleInitializationError, match="不支持的 LLM 适配器类型"):
            create_llm_adapter("unknown_type", "llm-mod", {})

    def test_adapter_inheritance(self):
        """验证继承关系"""
        adapter = create_llm_adapter("langchain", "llm-mod", {"api_key": "k"})
        assert isinstance(adapter, BaseLLM)
