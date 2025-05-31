import asyncio
import os
import logging  # 使用标准 logging
from typing import List, Dict, Any, Optional, AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage  # type: ignore
from langchain_core.language_models.chat_models import BaseChatModel  # type: ignore
from langchain_core.exceptions import OutputParserException  # type: ignore # 用于更具体的错误捕获

# 尝试导入特定提供商的Langchain类
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    ChatOpenAI = None
    logging.getLogger(__name__).warning("langchain_openai not installed. OpenAI provider will not be available.")

try:
    from langchain_deepseek import ChatDeepSeek  # type: ignore
except ImportError:  # pragma: no cover
    ChatDeepSeek = None
    logging.getLogger(__name__).warning("langchain_deepseek not installed. DeepSeek provider will not be available.")

# from langchain_anthropic import ChatAnthropic # 示例，如果需要支持

from core_framework.exceptions import ModuleInitializationError, ModuleProcessingError
from modules.base_llm import BaseLLM  # 继承自重构后的 BaseLLM

# from data_models.text_data import TextData # 虽然不直接用，但保持上下文一致性

logger = logging.getLogger(__name__)  # 使用标准 logging


class LangchainLLMAdapter(BaseLLM):
    """
    使用Langchain库与各种LLM提供商交互的适配器。
    """

    def __init__(self, module_id: str, config: Dict[str, Any],
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 event_manager: Optional[Any] = None,  # 'EventManager'
                 session_manager: Optional[Any] = None):  # 'SessionManager'
        """
        初始化 LangchainLLMAdapter。
        Args:
            module_id (str): 模块的唯一ID。
            config (Dict[str, Any]): LLM模块的完整配置字典。BaseLLM的__init__将处理它。
            event_loop, event_manager, session_manager: 由BaseLLM传递。
        """
        super().__init__(module_id, config, event_loop, event_manager, session_manager)

        self.llm_client: Optional[BaseChatModel] = None
        # 以下属性将从 self.provider_specific_config (由BaseLLM的__init__解析并设置) 中读取
        self.model_name: Optional[str] = None
        self.api_key: Optional[str] = None
        self.temperature: float = 0.7  # 默认值，会被配置覆盖

        logger.debug(f"LangchainLLMAdapter [{self.module_id}] __init__ 完成。等待 initialize() 调用。")

    async def initialize(self):
        """
        初始化Langchain LLM客户端。
        此方法在BaseLLM.initialize()之后被调用（如果BaseLLM.initialize()成功）。
        """


        if not self.enabled_provider_name:  # pragma: no cover (BaseLLM.initialize 应该已经捕获了这种情况)
            msg = f"LangchainLLMAdapter [{self.module_id}]: 'enable_module' (LLM提供商名称) 未在配置中指定。"
            logger.error(msg)
            self._is_ready = False
            raise ModuleInitializationError(msg)

        if not self.provider_specific_config:  # pragma: no cover (BaseLLM.initialize 应该已经捕获了这种情况)
            # 如果 enabled_provider_name 存在，但 provider_specific_config 为空字典，也可能是有问题的
            msg = f"LangchainLLMAdapter [{self.module_id}]: 未找到或无法加载提供商 '{self.enabled_provider_name}' 的特定配置。"
            logger.error(msg)
            self._is_ready = False
            raise ModuleInitializationError(msg)

        logger.info(
            f"LangchainLLMAdapter [{self.module_id}]: 开始初始化提供商 '{self.enabled_provider_name}' 的Langchain客户端...")

        # 从 self.provider_specific_config (已由BaseLLM的__init__填充) 中获取配置
        self.model_name = self.provider_specific_config.get("model_name")
        api_key_env_var = self.provider_specific_config.get("api_key_env_var")
        self.api_key = self.provider_specific_config.get("api_key")

        # 温度参数: 优先provider_specific_config, 其次全局module_config (self.config), 最后默认0.7
        self.temperature = float(self.provider_specific_config.get("temperature", self.config.get("temperature", 0.7)))

        if not self.model_name:
            msg = f"LangchainLLMAdapter [{self.module_id}]: 提供商 '{self.enabled_provider_name}' 的配置中缺少 'model_name'。"
            logger.error(msg)
            self._is_ready = False
            raise ModuleInitializationError(msg)

        # API Key 加载逻辑
        if not self.api_key and api_key_env_var:
            self.api_key = os.getenv(api_key_env_var)
            if self.api_key:
                logger.info(
                    f"LangchainLLMAdapter [{self.module_id}]: 已从环境变量 '{api_key_env_var}' 为提供商 '{self.enabled_provider_name}' 加载API密钥。")
            else:  # pragma: no cover
                logger.warning(f"LangchainLLMAdapter [{self.module_id}]: 环境变量 '{api_key_env_var}' 未设置或为空。")

        # 检查需要API Key的提供商
        providers_requiring_key = ["openai", "deepseek", "anthropic"]  # 根据实际支持的提供商调整
        if self.enabled_provider_name in providers_requiring_key and not self.api_key:
            msg = (f"LangchainLLMAdapter [{self.module_id}]: 提供商 '{self.enabled_provider_name}' 需要API密钥，"
                   f"但配置中未直接提供 'api_key'，且环境变量 '{api_key_env_var or '未指定'}' 未能提供密钥。")
            logger.error(msg)
            self._is_ready = False
            raise ModuleInitializationError(msg)

        # 初始化Langchain聊天模型客户端
        try:
            if self.enabled_provider_name == "openai":
                if not ChatOpenAI: raise ModuleInitializationError(
                    "ChatOpenAI (langchain_openai) 未安装。")  # pragma: no cover
                self.llm_client = ChatOpenAI(model=self.model_name, temperature=self.temperature,
                                             openai_api_key=self.api_key)  # type: ignore
            elif self.enabled_provider_name == "deepseek":
                if not ChatDeepSeek: raise ModuleInitializationError(
                    "ChatDeepSeek (langchain_deepseek) 未安装。")  # pragma: no cover
                # 确保传递正确的参数名，例如 deepseek_api_key
                self.llm_client = ChatDeepSeek(model=self.model_name, temperature=self.temperature,
                                               api_key=self.api_key)  # type: ignore
            # elif self.enabled_provider_name == "anthropic": # 示例
            #     if not ChatAnthropic: raise ModuleInitializationError("ChatAnthropic (langchain_anthropic) 未安装。")
            #     self.llm_client = ChatAnthropic(model=self.model_name, temperature=self.temperature, anthropic_api_key=self.api_key)
            else:
                msg = f"LangchainLLMAdapter [{self.module_id}]: 不支持的LLM提供商名称 '{self.enabled_provider_name}'。"
                logger.error(msg)
                self._is_ready = False
                raise ModuleInitializationError(msg)

            logger.info(
                f"LangchainLLMAdapter [{self.module_id}]: 已成功初始化提供商 '{self.enabled_provider_name}' 的模型 '{self.model_name}' (温度: {self.temperature})。")
            self._is_ready = True  # 只有在客户端成功初始化后才标记为就绪

        except Exception as e:  # pragma: no cover
            msg = f"LangchainLLMAdapter [{self.module_id}]: 初始化LLM客户端 (提供商: {self.enabled_provider_name}, 模型: {self.model_name}) 失败: {e}"
            logger.error(msg, exc_info=True)
            self._is_ready = False
            raise ModuleInitializationError(msg) from e

    def _prepare_llm_messages(self, user_prompt: str, dialogue_history: List[Dict[str, Any]]) -> List[BaseMessage]:
        """
        将用户提示和结构化的对话历史转换为Langchain BaseMessage对象列表。
        Args:
            user_prompt (str): 当前用户的提问。
            dialogue_history (List[Dict[str, Any]]): 格式化的历史对话列表，
                每条记录是一个包含 "role" 和 "content" 的字典。
                此历史记录不包含当前的用户提示。
        Returns:
            List[BaseMessage]: 准备好传递给LLM的消息对象列表。
        """
        messages: List[BaseMessage] = []  # type: ignore

        # 系统提示: 优先使用提供商特定配置中的system_prompt, 其次是模块全局的(self.config), 最后是BaseLLM的默认值(self.global_system_prompt)
        provider_system_prompt = self.provider_specific_config.get("system_prompt")
        # self.global_system_prompt 来自 BaseLLM.__init__
        final_system_prompt = provider_system_prompt if provider_system_prompt is not None else self.global_system_prompt

        if final_system_prompt and final_system_prompt.strip():
            messages.append(SystemMessage(content=final_system_prompt))  # type: ignore
            logger.debug(
                f"LangchainLLMAdapter [{self.module_id}]: 使用系统提示 (前50字符): '{final_system_prompt[:50]}...'")
        else:  # pragma: no cover
            logger.debug(f"LangchainLLMAdapter [{self.module_id}]: 未使用系统提示。")

        for entry in dialogue_history:  # dialogue_history 不包含当前 user_prompt
            role = entry.get("role")
            content = entry.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))  # type: ignore
            elif role == "assistant":
                messages.append(AIMessage(content=content))  # type: ignore
            # 可以根据需要扩展对其他角色 (如 "tool") 的处理
            # 例如: elif role == "tool": messages.append(ToolMessage(content=content, tool_call_id=entry.get("tool_call_id")))

        # 将当前用户的提问作为最后一条 HumanMessage
        messages.append(HumanMessage(content=user_prompt))  # type: ignore

        # logger.debug(f"LangchainLLMAdapter [{self.module_id}]: 准备了 {len(messages)} 条消息给LLM。最后一条是用户提示: '{user_prompt[:50]}...'")
        return messages

    async def _llm_ainvoke(self, messages: List[BaseMessage], session_id: str,
                           metadata: Optional[Dict[str, Any]]) -> AIMessage:
        """
        实现非流式LLM调用。
        """
        if not self.llm_client:  # pragma: no cover
            logger.error(f"LangchainLLMAdapter [{self.module_id}]: LLM客户端未初始化，无法执行ainvoke。")
            raise ModuleProcessingError(f"LLM客户端 (模块 {self.module_id}) 未初始化。")

        logger.debug(
            f"LangchainLLMAdapter [{self.module_id}] (提供商: {self.enabled_provider_name or '未知'}) 正在为会话 '{session_id}' 调用 ainvoke，消息数: {len(messages)}。")

        try:
            # langchain config 可以传递一些元数据或回调，具体看LLM提供商支持情况
            langchain_call_config = {"metadata": metadata or {},
                                     "tags": [f"session:{session_id}", f"module:{self.module_id}"]}
            response = await self.llm_client.ainvoke(messages, config=langchain_call_config)  # type: ignore

            if not isinstance(response, AIMessage):  # pragma: no cover
                logger.error(
                    f"LangchainLLMAdapter [{self.module_id}]: LLM ainvoke 返回了非AIMessage类型: {type(response)}。响应: {response}")
                raise ModuleProcessingError(f"LLM ainvoke 返回了意外的响应类型: {type(response)}")

            logger.debug(
                f"LangchainLLMAdapter [{self.module_id}] ainvoke 成功。响应内容 (前50字符): '{str(response.content)[:50]}...'")
            return response
        except OutputParserException as ope:  # pragma: no cover
            logger.error(f"LangchainLLMAdapter [{self.module_id}]: LLM ainvoke 期间发生输出解析错误: {ope}",
                         exc_info=True)
            raise ModuleProcessingError(f"LLM 输出解析错误: {ope}") from ope
        except Exception as e:  # pragma: no cover
            logger.error(f"LangchainLLMAdapter [{self.module_id}]: LLM ainvoke 期间发生未知错误: {e}", exc_info=True)
            raise ModuleProcessingError(f"LLM 调用 (ainvoke) 失败: {e}") from e

    async def _llm_astream(self, messages: List[BaseMessage], session_id: str, metadata: Optional[Dict[str, Any]]) -> \
    AsyncGenerator[AIMessage, None]:
        """
        实现流式LLM调用。
        """
        if not self.llm_client:  # pragma: no cover
            logger.error(f"LangchainLLMAdapter [{self.module_id}]: LLM客户端未初始化，无法执行astream。")
            raise ModuleProcessingError(f"LLM客户端 (模块 {self.module_id}) 未初始化。")

        logger.debug(
            f"LangchainLLMAdapter [{self.module_id}] (提供商: {self.enabled_provider_name or '未知'}) 正在为会话 '{session_id}' 调用 astream，消息数: {len(messages)}。")

        try:
            langchain_call_config = {"metadata": metadata or {},
                                     "tags": [f"session:{session_id}", f"module:{self.module_id}"]}
            async for chunk in self.llm_client.astream(messages, config=langchain_call_config):  # type: ignore
                if not isinstance(chunk, AIMessage):  # pragma: no cover
                    logger.warning(
                        f"LangchainLLMAdapter [{self.module_id}]: LLM astream 返回了非AIMessage类型的块: {type(chunk)}。块内容: {chunk}")
                    # 根据策略决定是跳过这个块还是作为错误处理
                    # 为保持流的连续性，如果块有content属性，尝试提取
                    if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                        yield AIMessage(content=chunk.content)  # type: ignore
                    continue

                # logger.debug(f"LangchainLLMAdapter [{self.module_id}] astream 产生块。内容 (前50字符): '{str(chunk.content)[:50]}...'")
                yield chunk
        except Exception as e:  # pragma: no cover
            logger.error(f"LangchainLLMAdapter [{self.module_id}]: LLM astream 期间发生错误: {e}", exc_info=True)
            raise ModuleProcessingError(f"LLM 流式调用 (astream) 失败: {e}") from e

    async def dispose(self):
        """
        清理LangchainLLMAdapter资源，例如关闭LLM客户端连接（如果支持）。
        """
        await super().dispose()  # 调用 BaseLLM.dispose()，它会重置 _is_ready 和 _is_initialized

        if self.llm_client and hasattr(self.llm_client, 'close') and callable(
                self.llm_client.close):  # pragma: no cover
            try:
                if asyncio.iscoroutinefunction(self.llm_client.close):  # type: ignore
                    await self.llm_client.close()  # type: ignore
                else:
                    self.llm_client.close()  # type: ignore
                logger.info(
                    f"LangchainLLMAdapter [{self.module_id}]: LLM客户端 ({self.enabled_provider_name or '未知'}) 资源已关闭/释放。")
            except Exception as e:
                logger.error(f"LangchainLLMAdapter [{self.module_id}]: 关闭LLM客户端时发生错误: {e}", exc_info=True)

        self.llm_client = None
        # _is_ready 和 _is_initialized 已经在 super().dispose() 中设置为 False
        logger.info(f"LangchainLLMAdapter [{self.module_id}] 已成功销毁。")
