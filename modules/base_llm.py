import asyncio
from abc import abstractmethod
import time
from typing import Optional, Dict, Any, List, AsyncGenerator

import logging # 切换到标准 logging

from data_models.text_data import TextData
from data_models.stream_event import StreamEvent, EventType
from modules.base_module import BaseModule
from core.session_manager import SessionManager
from core.exceptions import ModuleInitializationError, ModuleProcessingError
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class BaseLLM(BaseModule):
    """
    基础大语言模型 (LLM) 模块。
    负责与LLM适配器交互，处理用户输入，生成回复，
    管理聊天历史记录的传递，以及发出相关的流事件。
    支持流式和非流式两种响应模式。
    """

    def __init__(self, module_id: str, config: Dict[str, Any],
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 event_manager: Optional[Any] = None, # 'EventManager'
                 session_manager: Optional[SessionManager] = None):
        """
        初始化 BaseLLM 模块。

        Args:
            module_id (str): 模块的唯一ID。
            config (Dict[str, Any]): 此LLM模块的完整配置字典 (例如，来自config.yaml中modules.llm下的内容)。
            event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环。
            event_manager (Optional[Any]): 事件管理器实例。
            session_manager (Optional[SessionManager]): 会话管理器实例。
        """
        super().__init__(module_id, config, event_loop, event_manager)

        if not session_manager: # pragma: no cover
            logger.warning(f"BaseLLM [{self.module_id}]: SessionManager 未提供，历史记录功能将受限。")
        self.session_manager = session_manager

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
            logger.error(f"BaseLLM [{self.module_id}]: 配置中未找到 'enable_module' 字段。LLM模块无法确定要使用的提供商。")
            # 即使 enable_module 未找到，也继续初始化，但 is_ready 会是 False
            # 具体的适配器在 initialize 时会因为缺少 provider_specific_config 而失败
        else:
            all_provider_configs = self.config.get("config", {})
            if not isinstance(all_provider_configs, dict): # pragma: no cover
                logger.error(f"BaseLLM [{self.module_id}]: 模块配置中的 'config' 部分 (用于存放各提供商配置) 必须是一个字典，但找到的是 {type(all_provider_configs)}。")
            else:
                self.provider_specific_config = all_provider_configs.get(self.enabled_provider_name, {})
                if not self.provider_specific_config: # pragma: no cover
                    logger.warning(f"BaseLLM [{self.module_id}]: 在 'config' 部分中未找到已启用提供商 '{self.enabled_provider_name}' 的特定配置。将使用空配置。")

        logger.info(f"BaseLLM [{self.module_id}] 初始化。启用的提供商: '{self.enabled_provider_name or '未指定'}'. 流式输出: {self.streaming_enabled}. 最大历史Tokens: {self.max_history_tokens}.")
        logger.debug(f"BaseLLM [{self.module_id}] 全局系统提示 (前50字符): '{self.global_system_prompt[:50]}...'")
        if self.enabled_provider_name:
             logger.debug(f"BaseLLM [{self.module_id}] 提供商 '{self.enabled_provider_name}' 的特定配置: {self.provider_specific_config}")


    async def initialize(self):
        """
        基础LLM模块的初始化。
        主要检查基本配置是否齐全，具体的LLM客户端初始化由适配器子类完成。
        适配器子类在其 initialize 方法中应在成功时设置 self._is_ready = True。
        """
        if not self.enabled_provider_name: # pragma: no cover
            msg = f"BaseLLM [{self.module_id}]: 初始化失败，因为配置中缺少 'enable_module' 来指定LLM提供商。"
            logger.error(msg)
            self._is_initialized = False # 标记基础配置读取失败
            self._is_ready = False
            # 不在此处抛出 ModuleInitializationError，允许适配器在尝试初始化时再抛出
            return

        # BaseLLM 本身不做太多初始化，主要依赖适配器
        self._is_initialized = True # 标记基础配置已读取
        logger.debug(f"BaseLLM [{self.module_id}] 基础初始化完成。最终就绪状态取决于适配器。")

    @abstractmethod
    def _prepare_llm_messages(self, user_prompt: str, dialogue_history: List[Dict[str, Any]]) -> List[BaseMessage]:
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
    async def _llm_ainvoke(self, messages: List[BaseMessage], session_id: str, metadata: Optional[Dict[str, Any]]) -> AIMessage:
        """
        【适配器实现】以非流式方式调用LLM，返回单个AIMessage响应。
        Args:
            messages (List[BaseMessage]): 已准备好的LLM消息列表。
            session_id (str): 当前会话ID。
            metadata (Optional[Dict[str, Any]]): 传递给LLM调用的元数据。
        Returns:
            AIMessage: LLM的响应。
        Raises:
            ModuleProcessingError: 如果LLM调用失败或返回无效响应。
        """
        pass

    @abstractmethod
    async def _llm_astream(self, messages: List[BaseMessage], session_id: str, metadata: Optional[Dict[str, Any]]) -> AsyncGenerator[AIMessage, None]:
        """
        【适配器实现】以流式方式调用LLM，异步产生AIMessage块。
        Args:
            messages (List[BaseMessage]): 已准备好的LLM消息列表。
            session_id (str): 当前会话ID。
            metadata (Optional[Dict[str, Any]]): 传递给LLM调用的元数据。
        Yields:
            AIMessage: LLM响应的块。
        Raises:
            ModuleProcessingError: 如果LLM流式调用失败。
        """
        if False: # pragma: no cover
            yield AIMessage(content="") # type: ignore

    async def _prepare_llm_call_inputs(
            self,
            input_data: TextData,
            session_id: Optional[str]
    ) -> tuple[str, str, List[Dict[str, str]], Optional[Dict[str, Any]]]:
        """
        内部辅助方法：准备调用LLM所需的通用输入。
        返回: (会话ID, 用户提示文本, 对话历史列表, 原始输入元数据)
        """
        if not isinstance(input_data, TextData) or not input_data.text or not input_data.text.strip():
            error_msg = f"LLM模块 [{self.module_id}] 收到无效或空的 input_data (文本内容为空)，会话ID: {session_id}。"
            logger.warning(error_msg)
            raise ValueError(error_msg)

        # chunk_id 来自 TextData，通常由上游模块（如ASR或输入处理器）设置
        current_session_id = session_id or input_data.chunk_id or f"llm_sess_{int(time.time())}_{time.time_ns() % 1000}"
        prompt = input_data.text
        metadata = input_data.metadata # 从 TextData 中获取元数据

        logger.debug(f"LLM模块 [{self.module_id}] 正在为会话ID '{current_session_id}' 准备提示 (前100字符): '{prompt[:100]}...'")

        history: List[Dict[str, str]] = []
        if self.session_manager:
            # SessionManager.get_dialogue_history 返回 List[Dict[str, Any]]
            raw_history = self.session_manager.get_dialogue_history(current_session_id)
            # 转换历史记录格式，确保 'role' 和 'content' 是字符串
            for entry in raw_history:
                if isinstance(entry, dict) and isinstance(entry.get("role"), str) and isinstance(entry.get("content"), str):
                    history.append({"role": entry["role"], "content": entry["content"]})
                else: # pragma: no cover
                    logger.warning(f"LLM模块 [{self.module_id}] 会话 '{current_session_id}' 中的一条历史记录格式不符合预期: {entry}")
            logger.debug(f"LLM模块 [{self.module_id}] 为会话 '{current_session_id}' 获取并转换了 {len(history)} 条历史消息。")

        return current_session_id, prompt, history, metadata

    async def _handle_llm_success(
            self,
            response_text: str,
            prompt: str, # 原始用户提示
            session_id: str,
            source_metadata: Optional[Dict[str, Any]] # 来自原始TextData的元数据
    ) -> TextData:
        """
        内部辅助方法：处理LLM成功生成的响应。
        """
        logger.info( # 使用 info 级别记录成功响应的概要
            f"LLM模块 [{self.module_id}] 为会话 '{session_id}' 完成响应。完整文本 (前100字符): '{response_text[:100]}...'")

        final_response_data = TextData(
            text=response_text,
            chunk_id=session_id, # 使用会话ID作为 TextData 的 chunk_id
            is_final=True,
            # 合并元数据：模块信息、提供商信息、原始输入的元数据
            metadata={
                "source_prompt_start": prompt[:50], # 记录部分原始提示
                "module_id": self.module_id,
                "llm_provider": self.enabled_provider_name or "未知",
                **(source_metadata or {}) # 保留原始输入的元数据
            }
        )

        if self.event_manager:
            final_event = StreamEvent(
                event_type=EventType.SYSTEM_LLM_OUTPUT_FINAL, # 使用预定义的事件类型
                data=final_response_data,
                session_id=session_id, # StreamEvent 也携带 session_id
                metadata={"description": f"LLM final response from '{self.module_id}'."} # StreamEvent 的元数据
            )
            await self.event_manager.publish(final_event) # 假设是 publish

        if self.session_manager:
            # 将LLM的回复添加到历史记录
            self.session_manager.add_to_dialogue_history(session_id, {"role": "assistant", "content": response_text})
            logger.debug(f"LLM模块 [{self.module_id}] 已将助手回复添加到会话 '{session_id}' 的历史记录。")

        return final_response_data

    async def _handle_llm_error(self, error: Exception, prompt: str, session_id: str, pipeline_correlation_id: Optional[str]=None) -> None:
        """
        内部辅助方法：记录LLM处理过程中的错误并发出错误事件。
        """
        log_message = (
            f"LLM模块 [{self.module_id}] (提供商: {self.enabled_provider_name or '未知'}) "
            f"在为提示 (前100字符) '{prompt[:100]}...' (会话 '{session_id}') 处理时发生错误: {error}"
        )
        logger.error(log_message, exc_info=True) # exc_info=True 会记录完整的堆栈跟踪

        if self.event_manager:
            error_event_data = { # StreamEvent.data 字段的内容
                "module_id": self.module_id,
                "llm_provider": self.enabled_provider_name or "未知",
                "error_message": str(error),
                "original_prompt_start": prompt[:100], # 记录部分原始提示
                "session_id": session_id, # 在 data 内部也记录 session_id
            }
            if pipeline_correlation_id: # pragma: no cover
                 error_event_data["pipeline_correlation_id"] = pipeline_correlation_id

            error_event = StreamEvent(
                event_type=EventType.SYSTEM_MODULE_ERROR, # 使用预定义的错误事件类型
                data=error_event_data,
                session_id=session_id, # StreamEvent 也携带 session_id
                metadata={"description": f"LLM module '{self.module_id}' processing error: {error}"} # StreamEvent 的元数据
            )
            await self.event_manager.post_event(error_event)


    async def generate_complete_response(self, input_data: TextData, session_id: Optional[str] = None) -> Optional[TextData]:
        """
        以非流式方式调用LLM，获取一次性的完整回复。
        """
        if not self.is_ready: # pragma: no cover
            error_msg = f"LLM模块 [{self.module_id}] 在调用 generate_complete_response 时未就绪。"
            logger.error(error_msg)
            # 确保 session_id 和 input_data.text 有效以便记录错误
            effective_session_id = session_id or (input_data.chunk_id if input_data else f"err_sess_{time.time_ns()}")
            prompt_text_for_error = input_data.text if input_data and input_data.text else "未知提示"
            await self._handle_llm_error(ModuleProcessingError(error_msg), prompt_text_for_error, effective_session_id)
            return None

        try:
            current_session_id, prompt, history, source_metadata = await self._prepare_llm_call_inputs(input_data, session_id)
        except ValueError as e: # pragma: no cover
            logger.error(f"LLM模块 [{self.module_id}] 无法准备非流式输入的参数: {e}")
            # 尝试使用传入的 session_id 或 input_data.chunk_id 记录错误
            effective_session_id_for_error = session_id or (input_data.chunk_id if input_data else f"prep_err_sess_{time.time_ns()}")
            prompt_text_for_error_prep = input_data.text if input_data and input_data.text else "准备阶段的未知提示"
            await self._handle_llm_error(e, prompt_text_for_error_prep, effective_session_id_for_error)
            return None

        # 将用户当前消息添加到历史记录中，以便LLM调用时包含它
        if self.session_manager:
            self.session_manager.add_to_dialogue_history(current_session_id, {"role": "user", "content": prompt})
            logger.debug(f"LLM模块 [{self.module_id}] 已将用户输入添加到会话 '{current_session_id}' 的历史记录 (在调用LLM之前)。")
            # 更新 history 变量以包含当前用户输入，供 _prepare_llm_messages 使用
            history.append({"role": "user", "content": prompt})


        logger.info(f"LLM模块 [{self.module_id}] (提供商: {self.enabled_provider_name or '未知'}) 正在为会话 '{current_session_id}' 调用非流式响应。")

        try:
            # _prepare_llm_messages 由适配器实现，它会处理system_prompt的优先级和用户提示的添加
            # 注意：之前 history.append({"role": "user", "content": prompt}) 已将用户提示加入 history
            # _prepare_llm_messages 应该基于这个 history（现在包含用户提示）来构建消息
            # 或者，_prepare_llm_messages 的签名可以调整为 (self, user_prompt: str, dialogue_history_without_prompt: List[...])
            # 这里假设 _prepare_llm_messages 的 user_prompt 参数是当前的用户输入，而 dialogue_history 是不包含它的历史
            # 因此，从 history 中移除最后一条（即当前用户输入），并将其作为 user_prompt 参数传递
            if history and history[-1]["role"] == "user" and history[-1]["content"] == prompt:
                 history_for_prepare = history[:-1]
            else: # pragma: no cover (逻辑上应该总是匹配)
                 history_for_prepare = history
                 logger.warning(f"LLM模块 [{self.module_id}]: 历史记录的最后一条与当前用户提示不匹配，可能导致重复。")


            llm_messages = self._prepare_llm_messages(prompt, history_for_prepare) # prompt 是当前用户输入
            ai_message_response = await self._llm_ainvoke(llm_messages, current_session_id, source_metadata) # _llm_ainvoke 由适配器实现

            if ai_message_response and isinstance(ai_message_response.content, str) and ai_message_response.content.strip():
                response_text = ai_message_response.content
                return await self._handle_llm_success(response_text, prompt, current_session_id, source_metadata)
            else: # pragma: no cover
                logger.warning(f"LLM模块 [{self.module_id}] 的非流式响应无效或文本为空，会话 '{current_session_id}'。LLM原始回复: {ai_message_response}")
                await self._handle_llm_error(ModuleProcessingError("LLM returned empty or invalid non-streamed content"), prompt, current_session_id)
                return None
        except Exception as e: # pragma: no cover
            await self._handle_llm_error(e, prompt, current_session_id)
            return None

    async def stream_chat_response(self, input_data: TextData, session_id: Optional[str] = None) -> AsyncGenerator[TextData, None]:
        """
        以流式方式调用LLM，异步产生 (yield) 回复的各个部分 (chunks)。
        """
        if not self.is_ready: # pragma: no cover
            error_msg = f"LLM模块 [{self.module_id}] 在调用 stream_chat_response 时未就绪。"
            logger.error(error_msg)
            effective_session_id = session_id or (input_data.chunk_id if input_data else f"stream_err_sess_{time.time_ns()}")
            prompt_text_for_error = input_data.text if input_data and input_data.text else "未知提示（流式）"
            await self._handle_llm_error(ModuleProcessingError(error_msg), prompt_text_for_error, effective_session_id)
            # 对于异步生成器，可以yield一个表示错误的TextData或直接返回
            yield TextData(text="", chunk_id=effective_session_id, is_final=True, metadata={"error": error_msg, "module_id": self.module_id})
            return

        try:
            current_session_id, prompt, history, source_metadata = await self._prepare_llm_call_inputs(input_data, session_id)
        except ValueError as e: # pragma: no cover
            logger.error(f"LLM模块 [{self.module_id}] 无法准备流式输入的参数: {e}")
            effective_session_id_for_error = session_id or (input_data.chunk_id if input_data else f"stream_prep_err_sess_{time.time_ns()}")
            prompt_text_for_error_prep = input_data.text if input_data and input_data.text else "准备阶段的未知提示（流式）"
            await self._handle_llm_error(e, prompt_text_for_error_prep, effective_session_id_for_error)
            yield TextData(text="", chunk_id=effective_session_id_for_error, is_final=True, metadata={"error": str(e), "module_id": self.module_id})
            return

        # 将用户当前消息添加到历史记录
        if self.session_manager:
            self.session_manager.add_to_dialogue_history(current_session_id, {"role": "user", "content": prompt})
            logger.debug(f"LLM模块 [{self.module_id}] 已将用户输入添加到会话 '{current_session_id}' 的历史记录 (在流式调用LLM之前)。")
            # 更新 history 以包含当前用户输入
            history.append({"role": "user", "content": prompt})


        logger.info(f"LLM模块 [{self.module_id}] (提供商: {self.enabled_provider_name or '未知'}) 正在为会话 '{current_session_id}' 生成流式响应。")

        full_response_text = ""
        chunk_index = 0
        error_occurred = False
        final_response_generated = False # 标记是否已生成最终的TextData

        try:
            # 同样，从 history 中移除最后的用户提示，作为 user_prompt 参数传递
            if history and history[-1]["role"] == "user" and history[-1]["content"] == prompt:
                 history_for_prepare_stream = history[:-1]
            else: # pragma: no cover
                 history_for_prepare_stream = history

            llm_messages = self._prepare_llm_messages(prompt, history_for_prepare_stream)
            async for ai_message_chunk in self._llm_astream(llm_messages, current_session_id, source_metadata):
                if isinstance(ai_message_chunk, AIMessage) and isinstance(ai_message_chunk.content, str) and ai_message_chunk.content:
                    chunk_text = ai_message_chunk.content
                    full_response_text += chunk_text

                    partial_response_data = TextData(
                        text=chunk_text,
                        chunk_id=current_session_id, # 使用会话ID
                        is_final=False, # 流式块总是 is_final=False
                        metadata={
                            "chunk_index": chunk_index,
                            "module_id": self.module_id,
                            "llm_provider": self.enabled_provider_name or "未知",
                            **(source_metadata or {}) # 合并原始输入的元数据
                        }
                    )

                    if self.event_manager:
                        stream_event = StreamEvent(
                            event_type=EventType.SYSTEM_LLM_OUTPUT_CHUNK, # 使用预定义的事件类型
                            data=partial_response_data,
                            session_id=current_session_id, # StreamEvent 也携带 session_id
                            metadata={"description": f"LLM partial response chunk {chunk_index} from '{self.module_id}'."} # StreamEvent 的元数据
                        )
                        await self.event_manager.publish(stream_event)

                    yield partial_response_data
                    chunk_index += 1
                elif ai_message_chunk is None or not isinstance(ai_message_chunk, AIMessage) or not ai_message_chunk.content : # pragma: no cover (Adapter should yield AIMessage with content or end stream by stopping iteration)
                    logger.debug(f"LLM模块 [{self.module_id}] 流中收到空或无效块 (会话 '{current_session_id}'): {ai_message_chunk}")
                    # 如果适配器通过返回 None 来指示流结束，这里可以 break
                    # 但通常适配器会通过停止迭代来结束流

        except Exception as e: # pragma: no cover
            error_occurred = True
            await self._handle_llm_error(e, prompt, current_session_id)
            # 对于异步生成器，在发生错误后，也应该 yield 一个最终的、标记错误的 TextData
            yield TextData(text="", chunk_id=current_session_id, is_final=True, metadata={"error": str(e), "module_id": self.module_id, "status": "stream_error"})
            final_response_generated = True # 标记已生成错误响应

        # 流循环结束后 (或因错误中断后)
        if not final_response_generated: # 只有在没有错误，或者错误后没有生成最终响应时才执行
            if not error_occurred and full_response_text.strip():
                # 流成功结束且有内容，处理最终响应（包括历史记录更新和最终事件）
                # _handle_llm_success 内部会发送 SYSTEM_LLM_OUTPUT_FINAL 事件
                final_data = await self._handle_llm_success(full_response_text, prompt, current_session_id, source_metadata)
                # 流的消费者可能期望在流的最后接收到这个最终的 TextData 对象
                yield final_data
            elif not error_occurred and not full_response_text.strip(): # pragma: no cover
                logger.warning(f"LLM模块 [{self.module_id}] 的流式响应内容在结束后为空，会话 '{current_session_id}'。")
                # 仍然可以发送一个空的成功响应
                empty_final_data = await self._handle_llm_success("", prompt, current_session_id, source_metadata)
                yield empty_final_data
            elif error_occurred: # pragma: no cover (理论上错误时上面已经 yield 过了)
                 # 如果因为某些原因错误发生但没有 yield 最终错误 TextData
                 yield TextData(text="", chunk_id=current_session_id, is_final=True, metadata={"error": "Unknown stream error", "module_id": self.module_id, "status": "stream_error_unhandled"})


    async def process(self, input_data: TextData, session_id: Optional[str] = None, pipeline_correlation_id: Optional[str] = None) -> None:
        """
        LLM模块的主要处理方法。根据配置决定是使用流式还是非流式方式处理输入。
        这是模块被外部调用的主要入口点。
        此方法不直接返回响应，响应通过事件或流的消费者获取。
        """
        if not self.is_ready: # pragma: no cover
            error_msg = f"LLM模块 [{self.module_id}] 在调用 process 时未就绪。"
            logger.error(error_msg)
            effective_session_id = session_id or (input_data.chunk_id if input_data else f"proc_err_sess_{time.time_ns()}")
            prompt_text_for_error = input_data.text if input_data and input_data.text else "未知提示（process）"
            await self._handle_llm_error(ModuleProcessingError(error_msg), prompt_text_for_error, effective_session_id, pipeline_correlation_id)
            return

        prompt_for_logging = input_data.text if isinstance(input_data, TextData) else "N/A"
        # 确保 current_session_id_for_logging 有一个有效值
        current_session_id_for_logging = session_id or \
                                         (input_data.chunk_id if isinstance(input_data, TextData) and input_data.chunk_id else None) or \
                                         f"llm_proc_{int(time.time())}"


        processing_done_event_data = {
            "module_id": self.module_id,
            "session_id": current_session_id_for_logging,
            "prompt_text_start": prompt_for_logging[:50] if prompt_for_logging else "",
            "llm_provider": self.enabled_provider_name or "未知"
        }
        if pipeline_correlation_id: # pragma: no cover
            processing_done_event_data["pipeline_correlation_id"] = pipeline_correlation_id

        try:
            # 简单假设如果 streaming_enabled 为 True，则适配器应支持
            adapter_supports_streaming = self.streaming_enabled

            if self.streaming_enabled and adapter_supports_streaming:
                logger.debug(f"LLM模块 '{self.module_id}' 将使用流式处理会话 '{current_session_id_for_logging}'。")
                # 消耗异步生成器以确保所有操作（包括最终事件和历史更新）都执行
                async for _ in self.stream_chat_response(input_data, current_session_id_for_logging):
                    pass
                processing_done_event_data["status"] = "stream_completed"
            else:
                if not adapter_supports_streaming and self.streaming_enabled: # pragma: no cover
                    logger.warning(
                        f"LLM模块 '{self.module_id}' 配置为流式输出，但适配器 '{self.enabled_provider_name or '未知'}' 可能未完全支持或配置不当。将回退到非流式处理。"
                    )
                logger.debug(f"LLM模块 '{self.module_id}' 将使用非流式处理会话 '{current_session_id_for_logging}'。")
                await self.generate_complete_response(input_data, current_session_id_for_logging)
                processing_done_event_data["status"] = "non_stream_completed"

        except Exception as e: # pragma: no cover
            # 此处的异常通常是 _prepare_llm_call_inputs 或更早阶段的错误，
            # 因为 _llm_ainvoke 和 _llm_astream 内部的错误应该由它们自己处理并调用 _handle_llm_error
            logger.error(
                f"BaseLLM.process 中发生未处理的错误，提示 (前100字符): '{prompt_for_logging[:100]}...' (会话 '{current_session_id_for_logging}'): {e}",
                exc_info=True
            )
            processing_done_event_data["status"] = "error"
            processing_done_event_data["error_message"] = str(e)
            # 确保错误事件被发出
            await self._handle_llm_error(e, prompt_for_logging, current_session_id_for_logging, pipeline_correlation_id)
        finally:
            if self.event_manager:
                # 使用一个新事件类型或现有合适的事件类型来表示LLM处理完成
                llm_done_event_type = getattr(EventType, "SYSTEM_LLM_PROCESSING_COMPLETE", EventType.SYSTEM_MODULE_ERROR) # Fallback to a generic error if specific type not found

                processing_done_event = StreamEvent(
                    event_type=llm_done_event_type,
                    data=processing_done_event_data,
                    session_id=current_session_id_for_logging, # StreamEvent 也携带 session_id
                    metadata={"description": f"LLM module '{self.module_id}' has finished processing the prompt."} # StreamEvent 的元数据
                )
                await self.event_manager.publish(processing_done_event)


    async def start(self):
        """启动LLM模块。通常是被动调用，但可以用于预热等。"""
        if not self._is_initialized: # pragma: no cover
            logger.warning(f"LLM模块 '{self.module_id}' 尝试启动但尚未初始化。请先调用 initialize。")
            try:
                await self.initialize() # 尝试初始化
            except ModuleInitializationError as e_init_in_start:
                logger.error(f"LLM模块 '{self.module_id}' 在 start() 期间初始化失败: {e_init_in_start}。模块将不可用。")
                self._is_ready = False # 确保未就绪
                return # 初始化失败则不继续

        # 再次检查 is_ready，因为适配器的 initialize 可能会失败
        if not self.is_ready: # pragma: no cover
            logger.warning(f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 启动时仍未就绪。这可能表示适配器初始化存在问题。")
        else:
            logger.info(f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 已启动并就绪。")


    async def stop(self):
        """停止LLM模块并调用dispose清理资源。"""
        logger.info(f"LLM模块 '{self.module_id}' (提供商: {self.enabled_provider_name or '未知'}) 正在停止...")
        await self.dispose() # 调用dispose来清理资源
        logger.info(f"LLM模块 '{self.module_id}' 已停止。")

    async def dispose(self):
        """
        清理LLM模块资源。适配器应覆盖此方法以关闭其特定的LLM客户端。
        """
        logger.info(f"BaseLLM模块 '{self.module_id}' 正在销毁资源...")
        self._is_ready = False
        self._is_initialized = False # 销毁后也应重置初始化状态
        # 具体客户端的清理逻辑在适配器的 dispose 方法中 (通过 super().dispose() 调用)
        logger.info(f"BaseLLM模块 '{self.module_id}' 基础资源已标记为销毁。")
