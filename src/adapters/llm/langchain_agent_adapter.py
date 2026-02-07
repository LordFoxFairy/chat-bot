"""LangChain Agent 适配器

基于 LangChain 1.0+ create_agent API 实现。
继承自 LangChainLLMAdapter，复用其配置和初始化逻辑。
"""

from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
# from langchain.agents.middleware import AgentMiddleware

from src.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from src.core.models import TextData
from src.utils.logging_setup import logger
from src.adapters.llm.langchain_llm_adapter import LangChainLLMAdapter
from src.adapters.llm.middleware import (
    retry_middleware,
    log_before_model,
    log_after_model
)


class LangChainAgentAdapter(LangChainLLMAdapter):
    """LangChain Agent 适配器

    扩展 LangChainLLMAdapter，使用 create_agent 创建支持中间件的 Agent。

    配置扩充:
        middleware: List[str]  # 中间件列表: ["retry", "log_before", "log_after"]
        tools: List[str]       # 工具列表 (暂未实现具体加载逻辑)
    """

    MIDDLEWARE_MAP = {
        "retry": retry_middleware,
        "retry_middleware": retry_middleware,
        "log_before": log_before_model,
        "log_before_model": log_before_model,
        "log_after": log_after_model,
        "log_after_model": log_after_model,
    }

    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.agent = None
        self.middleware_names = self.config.get("middleware", [])
        self.tool_names = self.config.get("tools", [])

    async def _setup_impl(self):
        """初始化 Agent"""
        # 1. 初始化基础 LLM (ChatOpenAI) - 创建 self.llm
        await super()._setup_impl()

        logger.info(f"LLM/Agent [{self.module_id}] 正在初始化 Agent...")

        try:
            # 2. 准备中间件
            middleware_list = []
            for name in self.middleware_names:
                if mw := self.MIDDLEWARE_MAP.get(name):
                    middleware_list.append(mw)
                else:
                    logger.warning(f"LLM/Agent [{self.module_id}] 未知中间件: {name}")

            # 3. 准备工具 (目前仅占位，可后续扩展工具加载逻辑)
            tools_list = []
            if self.tool_names:
                logger.warning(f"LLM/Agent [{self.module_id}] 工具加载尚未实现，忽略配置: {self.tool_names}")

            # 4. 创建 Agent
            # 使用 LangChain 1.0+ create_agent API
            self.agent = create_agent(
                model=self.llm,  # 使用父类初始化的 ChatOpenAI 实例
                tools=tools_list,
                system_prompt=self.system_prompt,
                middleware=middleware_list
            )

            logger.info(f"LLM/Agent [{self.module_id}] Agent 初始化成功 (Middleware: {len(middleware_list)})")

        except Exception as e:
            logger.error(f"LLM/Agent [{self.module_id}] Agent 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"LangChain Agent 初始化失败: {e}") from e

    async def chat_stream(
        self,
        text: TextData,
        session_id: str
    ) -> AsyncGenerator[TextData, None]:
        """流式对话生成 (使用 Agent)"""
        if not self.agent:
            raise ModuleProcessingError("Agent 未初始化")

        if not text.text or not text.text.strip():
            logger.debug(f"LLM/Agent [{self.module_id}] 文本为空")
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                metadata={"status": "empty_input"}
            )
            return

        try:
            # 1. 管理会话历史 (复用父类逻辑维护 self.chat_histories)
            if session_id not in self.chat_histories:
                self.chat_histories[session_id] = [
                    SystemMessage(content=self.system_prompt)
                ]

            # 添加用户消息
            self.chat_histories[session_id].append(
                HumanMessage(content=text.text)
            )

            # 修剪历史
            self._trim_history(session_id)

            logger.debug(f"LLM/Agent [{self.module_id}] 开始流式生成")

            full_response = ""

            # 2. 准备 Agent 输入
            # 移除 SystemMessage，因为 create_agent 已配置 system_prompt
            messages_to_send = [
                m for m in self.chat_histories[session_id]
                if not isinstance(m, SystemMessage)
            ]

            # 3. 执行流式调用
            # version="v2" 用于获取标准化的 LangChain 事件
            async for event in self.agent.astream_events(
                {"messages": messages_to_send},
                version="v2"
            ):
                event_type = event.get("event")

                # 监听 Chat Model 输出事件
                # 注意: Agent 可能多次调用 Model (如工具链)，这里会输出所有生成的文本
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        full_response += content
                        yield TextData(
                            text=content,
                            chunk_id=session_id,
                            is_final=False,
                            metadata={"type": "chunk"}
                        )

            # 4. 更新历史记录 (添加助手回复)
            self.chat_histories[session_id].append(
                AIMessage(content=full_response)
            )

            logger.debug(f"LLM/Agent [{self.module_id}] 生成完成，长度: {len(full_response)}")

            # 发送结束标记
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                metadata={
                    "status": "complete",
                    "response_length": len(full_response)
                }
            )

        except Exception as e:
            # 异常处理
            logger.error(f"LLM/Agent [{self.module_id}] 生成失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"对话生成失败: {e}") from e

    async def close(self):
        """关闭资源"""
        self.agent = None
        await super().close()


def load() -> Type["LangChainAgentAdapter"]:
    """加载适配器类"""
    return LangChainAgentAdapter
