from typing import Dict, Any, AsyncGenerator, List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.exceptions import ModuleInitializationError, ModuleProcessingError
from data_models import TextData
from modules.base_llm import BaseLLM
from utils.logging_setup import logger


class LangChainLLMAdapter(BaseLLM):
    """LangChain LLM 适配器

    使用 LangChain 框架进行大语言模型对话。
    """

    # 默认配置常量
    DEFAULT_MODEL = "gpt-3.5-turbo"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取 LangChain 特定配置
        self.api_key = self.config.get("api_key")
        self.api_base = self.config.get("api_base")

        if not self.api_key:
            raise ModuleInitializationError("缺少 api_key 配置")

        # 对话历史管理（简单实现，每个 session 独立）
        self.chat_histories: Dict[str, List[BaseMessage]] = {}

        self.llm = None

        logger.info(f"LLM/LangChain [{self.module_id}] 配置加载完成:")
        logger.info(f"  - model: {self.model_name}")
        logger.info(f"  - temperature: {self.temperature}")
        logger.info(f"  - max_tokens: {self.max_tokens}")

    async def setup(self):
        """初始化 LangChain LLM"""
        logger.info(f"LLM/LangChain [{self.module_id}] 正在初始化...")

        try:
            # 初始化 LangChain ChatOpenAI
            config = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "api_key": self.api_key,
            }

            if self.api_base:
                config["base_url"] = self.api_base

            self.llm = ChatOpenAI(**config)

            self._is_initialized = True
            self._is_ready = True
            logger.info(f"LLM/LangChain [{self.module_id}] 初始化成功")

        except Exception as e:
            self._is_initialized = False
            self._is_ready = False
            logger.error(f"LLM/LangChain [{self.module_id}] 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"LangChain LLM 初始化失败: {e}") from e

    async def chat_stream(
        self,
        text: TextData,
        session_id: str
    ) -> AsyncGenerator[TextData, None]:
        """流式对话生成"""
        if not self.llm:
            raise ModuleProcessingError("LLM 未初始化")

        if not text.text or not text.text.strip():
            logger.debug(f"LLM/LangChain [{self.module_id}] 文本为空")
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                metadata={"status": "empty_input"}
            )
            return

        try:
            # 获取或初始化会话历史
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = [
                    SystemMessage(content=self.system_prompt)
                ]

            # 添加用户消息
            self.chat_histories[session_id].append(
                HumanMessage(content=text.text)
            )

            logger.debug(f"LLM/LangChain [{self.module_id}] 开始流式生成")

            # 流式调用
            full_response = ""
            async for chunk in self.llm.astream(self.chat_histories[session_id]):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield TextData(
                        text=chunk.content,
                        chunk_id=session_id,
                        is_final=False,
                        metadata={"type": "chunk"}
                    )

            # 添加助手回复到历史
            self.chat_histories[session_id].append(
                AIMessage(content=full_response)
            )

            logger.debug(f"LLM/LangChain [{self.module_id}] 生成完成，长度: {len(full_response)}")

            # 发送最终标记
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                metadata={"status": "complete", "response_length": len(full_response)}
            )

        except Exception as e:
            logger.error(f"LLM/LangChain [{self.module_id}] 生成失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"对话生成失败: {e}") from e

    async def close(self):
        """关闭 LLM，清理资源"""
        logger.info(f"LLM/LangChain [{self.module_id}] 正在关闭...")

        # 清理对话历史
        self.chat_histories.clear()
        self.llm = None

        self._is_ready = False
        self._is_initialized = False

        logger.info(f"LLM/LangChain [{self.module_id}] 已关闭")
        await super().close()
