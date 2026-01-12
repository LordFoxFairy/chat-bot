from abc import abstractmethod
from typing import Dict, Any, AsyncGenerator

from data_models import TextData
from modules.base_module import BaseModule
from utils.logging_setup import logger


class BaseLLM(BaseModule):
    """大语言模型模块基类

    职责:
    - 定义 LLM 核心接口
    - 提供通用的对话处理流程

    子类需要实现:
    - chat_stream: 流式对话生成
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取 LLM 通用配置
        self.model_name = self.config.get("model_name", "gpt-3.5-turbo")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.system_prompt = self.config.get("system_prompt", "You are a helpful AI assistant.")

        logger.debug(f"LLM [{self.module_id}] 配置加载:")
        logger.debug(f"  - model_name: {self.model_name}")
        logger.debug(f"  - temperature: {self.temperature}")
        logger.debug(f"  - max_tokens: {self.max_tokens}")

    @abstractmethod
    async def chat_stream(
        self,
        text: TextData,
        session_id: str
    ) -> AsyncGenerator[TextData, None]:
        """流式对话生成"""
        raise NotImplementedError("LLM 子类必须实现 chat_stream 方法")

    async def process_text(
        self,
        text: TextData,
        session_id: str
    ) -> AsyncGenerator[TextData, None]:
        """处理文本并生成回复流，内部调用 chat_stream"""
        if not self.is_ready:
            from core.exceptions import ModuleProcessingError
            raise ModuleProcessingError(f"LLM 模块 {self.module_id} 未就绪")

        try:
            logger.info(f"LLM [{self.module_id}] (Session: {session_id}) 开始对话: '{text.text[:30]}...'")

            async for response_chunk in self.chat_stream(text, session_id):
                yield response_chunk

            logger.info(f"LLM [{self.module_id}] (Session: {session_id}) 对话完成")

        except Exception as e:
            logger.error(
                f"LLM [{self.module_id}] (Session: {session_id}) 对话失败: {e}",
                exc_info=True
            )
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                metadata={
                    "error": str(e),
                    "source_module_id": self.module_id
                }
            )
