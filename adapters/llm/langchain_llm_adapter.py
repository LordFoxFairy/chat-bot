"""LangChain LLM 适配器

支持从环境变量读取 API Key，方便 LangChain ChatOpenAI 快速对接。
"""

import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from core.exceptions import ModuleInitializationError, ModuleProcessingError
from models import TextData
from modules.base_llm import BaseLLM
from utils.logging_setup import logger


class LangChainLLMAdapter(BaseLLM):
    """LangChain LLM 适配器

    使用 LangChain 框架进行大语言模型对话。
    支持从环境变量读取 API Key 和 Base URL。

    配置示例:
        model_name: "gpt-3.5-turbo"
        api_key_env_var: "OPENAI_API_KEY"  # 从环境变量读取
        base_url: "https://api.openai.com/v1"  # 可选
    """

    # 会话历史配置常量
    MAX_HISTORY_LENGTH = 20
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 解析 API Key（支持环境变量）
        self.api_key = self._resolve_api_key()
        self.api_base = self._resolve_base_url()

        # 其他配置
        self.max_history_length = self.config.get("max_history_length", self.MAX_HISTORY_LENGTH)
        self.max_retries = self.config.get("max_retries", self.MAX_RETRIES)
        self.retry_delay = self.config.get("retry_delay", self.RETRY_DELAY)

        # 对话历史管理
        self.chat_histories: Dict[str, List[BaseMessage]] = {}
        self.llm: Optional[ChatOpenAI] = None

        logger.info(f"LLM/LangChain [{self.module_id}] 配置加载完成:")
        logger.info(f"  - model: {self.model_name}")
        logger.info(f"  - base_url: {self.api_base or 'default'}")
        logger.info(f"  - temperature: {self.temperature}")
        logger.info(f"  - max_tokens: {self.max_tokens}")

    def _resolve_api_key(self) -> str:
        """解析 API Key（环境变量优先）

        优先级: 环境变量 > 配置文件

        Returns:
            API Key 字符串

        Raises:
            ModuleInitializationError: 当无法获取 API Key 时
        """
        # 1. 尝试从环境变量获取
        api_key_env = self.config.get("api_key_env_var")
        if api_key_env:
            env_value = os.getenv(api_key_env)
            if env_value:
                logger.debug(f"LLM [{self.module_id}] 从环境变量 '{api_key_env}' 读取 API Key")
                return env_value
            logger.warning(f"LLM [{self.module_id}] 环境变量 '{api_key_env}' 未设置")

        # 2. 尝试从配置文件获取
        config_key = self.config.get("api_key")
        if config_key:
            logger.warning(f"LLM [{self.module_id}] 从配置文件读取 API Key（建议使用环境变量）")
            return config_key

        # 3. 抛出异常
        raise ModuleInitializationError(
            f"缺少 API Key，请设置环境变量 '{api_key_env}' 或在配置中提供 'api_key'"
        )

    def _resolve_base_url(self) -> Optional[str]:
        """解析 Base URL（环境变量优先）

        Returns:
            Base URL 或 None
        """
        # 1. 尝试从环境变量获取
        base_url_env = self.config.get("base_url_env_var")
        if base_url_env:
            env_value = os.getenv(base_url_env)
            if env_value:
                logger.debug(f"LLM [{self.module_id}] 从环境变量 '{base_url_env}' 读取 Base URL")
                return env_value

        # 2. 从配置文件获取
        return self.config.get("base_url") or self.config.get("api_base")

    async def _setup_impl(self):
        """初始化 LangChain LLM (内部实现)"""
        logger.info(f"LLM/LangChain [{self.module_id}] 正在初始化...")

        try:
            # 初始化 LangChain ChatOpenAI
            init_config: Dict[str, Any] = {
                "model": self.model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "api_key": self.api_key,
            }

            if self.api_base:
                init_config["base_url"] = self.api_base

            self.llm = ChatOpenAI(**init_config)

            logger.info(f"LLM/LangChain [{self.module_id}] 初始化成功")

        except Exception as e:
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

            # 限制历史长度
            self._trim_history(session_id)

            logger.debug(f"LLM/LangChain [{self.module_id}] 开始流式生成")

            # 流式调用（带重试）
            full_response = ""
            retry_count = 0

            while retry_count <= self.max_retries:
                try:
                    async for chunk in self.llm.astream(self.chat_histories[session_id]):
                        if hasattr(chunk, "content") and chunk.content:
                            full_response += chunk.content
                            yield TextData(
                                text=chunk.content,
                                chunk_id=session_id,
                                is_final=False,
                                metadata={"type": "chunk"}
                            )
                    break

                except Exception as stream_error:
                    retry_count += 1
                    error_type = type(stream_error).__name__

                    if retry_count <= self.max_retries:
                        logger.warning(
                            f"LLM/LangChain [{self.module_id}] 流式生成失败 ({error_type}), "
                            f"重试 {retry_count}/{self.max_retries}..."
                        )
                        await asyncio.sleep(self.retry_delay * retry_count)
                    else:
                        logger.error(
                            f"LLM/LangChain [{self.module_id}] 流式生成失败，已达最大重试次数: {stream_error}",
                            exc_info=True
                        )
                        raise ModuleProcessingError(
                            f"对话生成失败（已重试{self.max_retries}次）: {stream_error}"
                        ) from stream_error

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

        except ModuleProcessingError:
            raise
        except Exception as e:
            logger.error(f"LLM/LangChain [{self.module_id}] 生成失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"对话生成失败: {e}") from e

    def _trim_history(self, session_id: str):
        """修剪会话历史，保持在最大长度限制内"""
        history = self.chat_histories[session_id]

        if len(history) <= self.max_history_length:
            return

        # 确保第一条是 SystemMessage
        system_msg = history[0]
        if not isinstance(system_msg, SystemMessage):
            # 如果第一条不是 SystemMessage，创建一个新的
            system_msg = SystemMessage(content=self.system_prompt)

        recent_msgs = history[-(self.max_history_length - 1):]
        self.chat_histories[session_id] = [system_msg] + recent_msgs

        logger.debug(
            f"LLM/LangChain [{self.module_id}] 会话 {session_id} 历史已修剪至 "
            f"{len(self.chat_histories[session_id])} 条"
        )

    def clear_history(self, session_id: str):
        """清除指定会话的历史记录"""
        if session_id in self.chat_histories:
            self.chat_histories.pop(session_id)
            logger.info(f"LLM/LangChain [{self.module_id}] 清除会话历史: {session_id}")

    def get_history_length(self, session_id: str) -> int:
        """获取指定会话的历史记录长度"""
        if session_id in self.chat_histories:
            return len(self.chat_histories[session_id])
        return 0

    async def close(self):
        """关闭 LLM，清理资源"""
        logger.info(f"LLM/LangChain [{self.module_id}] 正在关闭...")

        self.chat_histories.clear()
        self.llm = None

        logger.info(f"LLM/LangChain [{self.module_id}] 已关闭")
        await super().close()


def load() -> Type["LangChainLLMAdapter"]:
    """加载 LangChainLLM 适配器类"""
    return LangChainLLMAdapter
