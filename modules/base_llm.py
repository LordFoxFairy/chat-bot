import asyncio
from abc import abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator

from langchain_core.messages import AIMessage, BaseMessage

from core.exceptions import ModuleInitializationError
from data_models.text_data import TextData
from modules.base_module import BaseModule
from utils.logging_setup import logger


class BaseLLM(BaseModule):
    """
    基础大语言模型 (LLM) 模块。
    负责与LLM适配器交互，处理用户输入，生成回复，
    管理聊天历史记录的传递，以及发出相关的流事件。
    支持流式和非流式两种响应模式。
    """

    def __init__(self, module_id: str, config: Dict[str, Any],
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化 BaseLLM 模块。

        Args:
            module_id (str): 模块的唯一ID。
            config (Dict[str, Any]): 此LLM模块的完整配置字典 (例如，来自config.yaml中modules.llm下的内容)。
            event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
        """
        super().__init__(module_id, config, event_loop)

        # 从模块的完整配置 (self.config) 中加载通用LLM设置
        self.streaming_enabled: bool = self.config.get("streaming_enabled", True)
        # 注意: token计数依赖适配器, BaseLLM 不直接处理 token 数量，但保留配置项
        self.max_history_tokens: int = self.config.get("max_history_tokens", 2000)
        self.global_system_prompt: str = self.config.get("system_prompt", "You are a helpful AI assistant.")

        # 确定当前启用的LLM提供商及其特定配置
        # 'enable_module' 字段指定了在 'config' 子字典中要使用的提供商配置块的键名
        self.enabled_provider_name: Optional[str] = self.config.get("enable_module")
        self.provider_specific_config: Dict[str, Any] = {}

        if not self.enabled_provider_name:
            logger.error(
                f"BaseLLM [{self.module_id}]: 配置中未找到 'enable_module' 字段。LLM模块无法确定要使用的提供商。")
            # 即使 enable_module 未找到，也继续初始化，但 is_ready 会是 False
            # 具体的适配器在 initialize 时会因为缺少 provider_specific_config 而失败
        else:
            all_provider_configs = self.config.get("config", {})
            if not isinstance(all_provider_configs, dict):  # pragma: no cover
                logger.error(
                    f"BaseLLM [{self.module_id}]: 模块配置中的 'config' 部分 (用于存放各提供商配置) 必须是一个字典，但找到的是 {type(all_provider_configs)}。")
            else:
                self.provider_specific_config = all_provider_configs.get(self.enabled_provider_name, {})
                if not self.provider_specific_config:  # pragma: no cover
                    logger.warning(
                        f"BaseLLM [{self.module_id}]: 在 'config' 部分中未找到已启用提供商 '{self.enabled_provider_name}' 的特定配置。将使用空配置。")

        logger.info(
            f"BaseLLM [{self.module_id}] 初始化。启用的提供商: '{self.enabled_provider_name or '未指定'}'. 流式输出: {self.streaming_enabled}. 最大历史Tokens: {self.max_history_tokens}.")
        logger.debug(f"BaseLLM [{self.module_id}] 全局系统提示 (前50字符): '{self.global_system_prompt[:50]}...'")
        if self.enabled_provider_name:
            logger.debug(
                f"BaseLLM [{self.module_id}] 提供商 '{self.enabled_provider_name}' 的特定配置: {self.provider_specific_config}")

    async def initialize(self):
        """
        基础LLM模块的初始化。
        主要检查基本配置是否齐全，具体的LLM客户端初始化由适配器子类完成。
        适配器子类在其 initialize 方法中应在成功时设置 self._is_ready = True。
        """
        # BaseModule.initialize() 通常是pass, 但如果它有逻辑，应该调用
        # await super().initialize() # 如果 BaseModule 有异步 initialize

        if not self.enabled_provider_name:  # pragma: no cover
            msg = f"BaseLLM [{self.module_id}]: 初始化失败，因为配置中缺少 'enable_module' 来指定LLM提供商。"
            logger.error(msg)
            self._is_initialized = False  # 标记基础配置读取失败
            self._is_ready = False
            # 不在此处抛出 ModuleInitializationError，允许适配器在尝试初始化时再抛出
            return

        # BaseLLM 本身不做太多初始化，主要依赖适配器
        # _is_initialized 标记配置已读取, _is_ready 由适配器在成功连接LLM后设置
        self._is_initialized = True  # 标记基础配置已读取
        logger.debug(f"BaseLLM [{self.module_id}] 基础初始化完成。最终就绪状态取决于适配器。")
        # self._is_ready 将由具体的适配器子类在其initialize方法中设置为True

    @abstractmethod
    def _prepare_llm_messages(self, user_prompt: str, dialogue_history: Optional[List[Dict[str, Any]]]) -> List[
        BaseMessage]:
        """
        【适配器实现】将用户提示和对话历史转换为LLM期望的消息格式列表。
        此方法应负责处理系统提示的优先级（提供商特定 > 模块全局 > BaseLLM默认）。
        Args:
            user_prompt (str): 当前用户的提问。
            dialogue_history (List[Dict[str, Any]]): 格式化的历史对话列表，
                每条记录是一个包含 "role" 和 "content" 的字典。
                注意：此历史记录不应包含当前的 user_prompt。
        Returns:
            List[BaseMessage]: 准备好传递给LLM的消息对象列表。
        """
        pass

    @abstractmethod
    async def _llm_ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        【适配器实现】以非流式方式调用LLM，返回单个AIMessage响应。
        Args:
            messages (List[BaseMessage]): 已准备好的LLM消息列表。
        Returns:
            AIMessage: LLM的响应。
        Raises:
            ModuleProcessingError: 如果LLM调用失败或返回无效响应。
        """
        pass

    @abstractmethod
    async def _llm_astream(self, messages: List[BaseMessage]) -> \
            AsyncGenerator[AIMessage, None]:
        """
        【适配器实现】以流式方式调用LLM，异步产生AIMessage块。
        Args:
            messages (List[BaseMessage]): 已准备好的LLM消息列表。
        Yields:
            AIMessage: LLM响应的块。
        Raises:
            ModuleProcessingError: 如果LLM流式调用失败。
        """
        if False:
            yield AIMessage(content="")

    async def _handle_llm_success(self, response_text: str, prompt: str) -> TextData:
        """
        内部辅助方法：处理LLM成功生成的响应。
        """
        final_response_data = TextData(text=response_text, is_final=True, )
        return final_response_data

    async def generate_complete_response(self, input_data: TextData) -> Optional[TextData]:
        """
        以非流式方式调用LLM，获取一次性的完整回复。
        """

        try:
            prompt = input_data.text
        except ValueError as e:  # pragma: no cover
            logger.error(f"LLM模块 [{self.module_id}] 无法准备非流式输入的参数: {e}")
            return None

        try:

            llm_messages = self._prepare_llm_messages(prompt, [])  # prompt 是当前用户输入
            ai_message_response = await self._llm_ainvoke(llm_messages)

            if ai_message_response and isinstance(ai_message_response.content,
                                                  str) and ai_message_response.content.strip():
                response_text = ai_message_response.content
                return await self._handle_llm_success(response_text, prompt)
        except Exception as e:
            return None

    async def stream_chat_response(self, input_data: TextData, session_id: Optional[str] = None) -> AsyncGenerator[
        str, None]:
        """
        以流式方式调用LLM，异步产生 (yield) 回复的各个部分 (chunks)。
        """
        prompt = input_data.text
        chunk_index = 0

        llm_messages = self._prepare_llm_messages(prompt, [])
        async for ai_message_chunk in self._llm_astream(llm_messages):
            if isinstance(ai_message_chunk, AIMessage) and \
                    isinstance(ai_message_chunk.content, str) and ai_message_chunk.content:
                chunk_text = ai_message_chunk.content
                yield chunk_text
                chunk_index += 1

    async def start(self):
        """启动LLM模块。通常是被动调用，但可以用于预热等。"""
        if not self._is_initialized:  # pragma: no cover
            logger.warning(f"LLM模块 '{self.module_id}' 尝试启动但尚未初始化。请先调用 initialize。")
            try:
                await self.initialize()  # 尝试初始化
            except ModuleInitializationError as e_init_in_start:
                logger.error(f"LLM模块 '{self.module_id}' 在 start() 期间初始化失败: {e_init_in_start}。模块将不可用。")
                self._is_ready = False  # 确保未就绪
                return  # 初始化失败则不继续

        # 再次检查 is_ready，因为适配器的 initialize 可能会失败
        if not self.is_ready:  # pragma: no cover
            logger.warning(
                f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 启动时仍未就绪。这可能表示适配器初始化存在问题。")
        else:
            logger.info(f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 已启动并就绪。")

    async def stop(self):
        """停止LLM模块并调用dispose清理资源。"""
        logger.info(f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 正在停止...")
        await self.dispose()  # 调用dispose来清理资源
        logger.info(f"LLM模块 '{self.module_id}' 已停止。")

    async def dispose(self):
        """
        清理LLM模块资源。适配器应覆盖此方法以关闭其特定的LLM客户端。
        """
        logger.info(f"BaseLLM模块 '{self.module_id}' 正在销毁资源...")
        self._is_ready = False
        self._is_initialized = False  # 销毁后也应重置初始化状态
        # 具体客户端的清理逻辑在适配器的 dispose 方法中 (通过 super().dispose() 调用)
        logger.info(f"BaseLLM模块 '{self.module_id}' 基础资源已标记为销毁。")
