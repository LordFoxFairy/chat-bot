from typing import Callable, Awaitable, Optional, Dict

from core.session_context import SessionContext
from models import StreamEvent, EventType, TextData
from utils.logging_setup import logger


class TextInputHandler:
    """文本输入处理器

    职责:
    - 接收文本输入
    - 清洗和验证文本
    - 发送处理结果

    特点:
    - 与 AudioInputHandler 对称设计
    - 全异步
    """

    def __init__(
        self,
        session_context: SessionContext,
        result_callback: Callable[[StreamEvent, dict], Awaitable[None]]
    ):
        self.session_context = session_context
        self.result_callback = result_callback

        logger.info(f"[TextInput] Initialized for session {self.session_context.session_id}")

    async def process_text(self, text: str):
        """处理文本输入

        Args:
            text: 用户输入的文本
        """
        # 清洗文本
        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            logger.warning(f"[TextInput] Empty text after cleaning, session={self.session_context.session_id}")
            return

        logger.info(f"[TextInput] Processing text: '{cleaned_text}', session={self.session_context.session_id}")

        # 发送文本结果事件（类似 ASR_RESULT）
        event = StreamEvent(
            event_type=EventType.ASR_RESULT,  # 复用 ASR_RESULT，因为都是文本输入的最终结果
            event_data=TextData(
                text=cleaned_text,
                is_final=True
            ),
            session_id=self.session_context.session_id,
            tag_id=self.session_context.tag_id
        )

        await self.result_callback(event, {"session_id": self.session_context.session_id})

    def _clean_text(self, text: str) -> str:
        """清洗文本

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text:
            return ""

        # 去除首尾空白
        cleaned = text.strip()

        # 未来可以添加更多清洗逻辑：
        # - 过滤敏感词
        # - 标准化标点符号
        # - 移除特殊字符

        return cleaned
