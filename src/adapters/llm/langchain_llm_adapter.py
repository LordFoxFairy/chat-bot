"""LangChain LLM 适配器

支持多种模型提供商（OpenAI、Anthropic、Google 等），
通过三个核心参数配置：model、api_key、base_url。
使用 langchain 的 init_chat_model 自动选择正确的客户端。
"""

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.core.interfaces.base_llm import BaseLLM
from src.core.models import TextData
from src.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from src.utils.logging_setup import logger


class LangChainLLMAdapter(BaseLLM):
    """LangChain LLM 适配器

    使用三个核心参数配置 LLM：
    - model_name: 模型名称（如 claude-sonnet-4-20250514, gpt-4, anthropic/claude-3-opus）
    - api_key / api_key_env_var: API 密钥
    - base_url: API 端点（可选，用于 OpenRouter 等第三方服务）

    配置示例:
        # OpenRouter (推荐)
        model_name: "anthropic/claude-3-opus"
        api_key_env_var: "OPENROUTER_API_KEY"
        base_url: "https://openrouter.ai/api/v1"

        # Anthropic 直连
        model_name: "claude-sonnet-4-20250514"
        api_key_env_var: "ANTHROPIC_API_KEY"

        # OpenAI 直连
        model_name: "gpt-4"
        api_key_env_var: "OPENAI_API_KEY"
    """

    MAX_HISTORY_LENGTH = 20
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)

        # 核心三要素
        self.api_key = self._resolve_api_key()
        self.base_url = self._resolve_base_url()

        # 历史管理配置
        self.max_history_length = config.get("max_history_length", self.MAX_HISTORY_LENGTH)
        self.max_retries = config.get("max_retries", self.MAX_RETRIES)
        self.retry_delay = config.get("retry_delay", self.RETRY_DELAY)

        # 运行时状态
        self.chat_histories: Dict[str, List[BaseMessage]] = {}
        self.llm: Optional[BaseChatModel] = None

        logger.info(f"LLM [{self.module_id}] 配置:")
        logger.info(f"  - model: {self.model_name}")
        logger.info(f"  - base_url: {self.base_url or 'default'}")
        logger.info(f"  - temperature: {self.temperature}")

    def _resolve_api_key(self) -> str:
        """解析 API Key（环境变量优先）"""
        # 从环境变量读取
        if api_key_env := self.config.get("api_key_env_var"):
            if env_value := os.getenv(api_key_env):
                logger.debug(f"LLM [{self.module_id}] 从环境变量 '{api_key_env}' 读取 API Key")
                return env_value
            logger.warning(f"LLM [{self.module_id}] 环境变量 '{api_key_env}' 未设置")

        # 从配置读取
        if config_key := self.config.get("api_key"):
            return config_key

        raise ModuleInitializationError(
            f"缺少 API Key，请设置环境变量或在配置中提供 'api_key'"
        )

    def _resolve_base_url(self) -> Optional[str]:
        """解析 Base URL"""
        if base_url_env := self.config.get("base_url_env_var"):
            if env_value := os.getenv(base_url_env):
                return env_value
        return self.config.get("base_url") or self.config.get("api_base")

    def _init_model(self) -> BaseChatModel:
        """初始化 LLM 模型

        使用 init_chat_model 自动选择正确的客户端，
        或根据 base_url 使用 OpenAI 兼容模式。
        """
        kwargs: Dict[str, Any] = {
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens

        # 如果有 base_url，使用 ChatOpenAI（OpenAI 兼容模式，适用于 OpenRouter 等）
        if self.base_url:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        # 没有 base_url，尝试使用 init_chat_model 自动选择
        try:
            from langchain.chat_models import init_chat_model

            return init_chat_model(
                model=self.model_name,
                api_key=self.api_key,
                **kwargs,
            )
        except (ImportError, Exception) as e:
            logger.debug(f"init_chat_model 失败: {e}，尝试其他方式")

        # 根据模型名称前缀选择客户端
        model_lower = self.model_name.lower()

        if model_lower.startswith("claude"):
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=self.model_name,
                api_key=self.api_key,
                **kwargs,
            )

        # 默认使用 ChatOpenAI
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            **kwargs,
        )

    async def _setup_impl(self) -> None:
        """初始化 LLM"""
        logger.info(f"LLM [{self.module_id}] 正在初始化...")

        try:
            self.llm = self._init_model()
            logger.info(f"LLM [{self.module_id}] 初始化成功")
        except Exception as e:
            logger.error(f"LLM [{self.module_id}] 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"LLM 初始化失败: {e}") from e

    async def chat_stream(
        self,
        text: TextData,
        session_id: str,
    ) -> AsyncGenerator[TextData, None]:
        """流式对话生成"""
        if not self.llm:
            raise ModuleProcessingError("LLM 未初始化")

        if not text.text or not text.text.strip():
            yield TextData(text="", chunk_id=session_id, is_final=True)
            return

        try:
            # 初始化会话历史
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = [SystemMessage(content=self.system_prompt)]

            self.chat_histories[session_id].append(HumanMessage(content=text.text))
            self._trim_history(session_id)

            # 流式生成（带重试）
            full_response = ""
            for retry in range(self.max_retries + 1):
                try:
                    async for chunk in self.llm.astream(self.chat_histories[session_id]):
                        if hasattr(chunk, "content") and chunk.content:
                            full_response += chunk.content
                            yield TextData(
                                text=chunk.content,
                                chunk_id=session_id,
                                is_final=False,
                            )
                    break
                except Exception as e:
                    if retry < self.max_retries:
                        logger.warning(f"LLM [{self.module_id}] 重试 {retry + 1}/{self.max_retries}: {e}")
                        await asyncio.sleep(self.retry_delay * (retry + 1))
                    else:
                        raise ModuleProcessingError(f"生成失败: {e}") from e

            self.chat_histories[session_id].append(AIMessage(content=full_response))
            yield TextData(text="", chunk_id=session_id, is_final=True)

        except ModuleProcessingError:
            raise
        except Exception as e:
            logger.error(f"LLM [{self.module_id}] 生成失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"生成失败: {e}") from e

    def _trim_history(self, session_id: str) -> None:
        """修剪会话历史"""
        history = self.chat_histories[session_id]
        if len(history) <= self.max_history_length:
            return

        system_msg = history[0] if isinstance(history[0], SystemMessage) else SystemMessage(content=self.system_prompt)
        recent = history[-(self.max_history_length - 1):]
        self.chat_histories[session_id] = [system_msg] + recent

    def clear_history(self, session_id: str) -> None:
        """清除会话历史"""
        self.chat_histories.pop(session_id, None)

    def get_history_length(self, session_id: str) -> int:
        """获取会话历史长度"""
        return len(self.chat_histories.get(session_id, []))

    async def close(self) -> None:
        """关闭 LLM"""
        self.chat_histories.clear()
        self.llm = None
        await super().close()


def load() -> Type[LangChainLLMAdapter]:
    """加载适配器类"""
    return LangChainLLMAdapter
