import asyncio
from typing import Dict, Optional, Callable, Awaitable

from src.core.models import StreamEvent
from src.utils.logging_setup import logger
from src.core.conversation.orchestrator import ConversationOrchestrator
from .session_manager import SessionManager


class ConversationManager:
    """会话管理器

    职责:
    - 管理所有 ConversationHandler 的生命周期
    - 提供会话创建、获取、销毁接口
    - 不涉及协议传输（由 Protocol 负责）
    - 不涉及模块管理（由 ChatEngine 负责）
    """

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.conversation_handlers: Dict[str, ConversationOrchestrator] = {}
        self._lock = asyncio.Lock()
        logger.info("ConversationManager 初始化完成")

    async def create_conversation_handler(
        self,
        session_id: str,
        tag_id: str,
        send_callback: Callable[[StreamEvent], Awaitable[None]],
        session_context: 'SessionContext'
    ) -> ConversationOrchestrator:
        """创建并启动 ConversationOrchestrator

        Args:
            session_id: 会话唯一标识
            tag_id: 客户端标签
            send_callback: 消息发送回调函数
            session_context: 会话上下文（包含 engine 引用）

        Returns:
            ConversationOrchestrator: 创建的会话处理器
        """
        async with self._lock:
            if session_id in self.conversation_handlers:
                logger.warning(f"ConversationManager 会话已存在: {session_id}")
                return self.conversation_handlers[session_id]

            handler = ConversationOrchestrator(
                session_id=session_id,
                tag_id=tag_id,
                session_context=session_context,
                session_manager=self.session_manager,
                send_callback=send_callback
            )
            await handler.start()

            self.conversation_handlers[session_id] = handler
            logger.info(f"ConversationManager 创建会话: session={session_id}, tag={tag_id}")

            return handler

    def get_conversation_handler(self, session_id: str) -> Optional[ConversationOrchestrator]:
        """获取会话处理器

        Args:
            session_id: 会话唯一标识

        Returns:
            Optional[ConversationOrchestrator]: 会话处理器，不存在则返回 None
        """
        return self.conversation_handlers.get(session_id)

    async def destroy_conversation_handler(self, session_id: str):
        """销毁会话处理器

        Args:
            session_id: 会话唯一标识
        """
        async with self._lock:
            handler = self.conversation_handlers.pop(session_id, None)

        if handler:
            await handler.stop()
            logger.info(f"ConversationManager 销毁会话: session={session_id}")
        else:
            logger.warning(f"ConversationManager 会话不存在: {session_id}")

    async def destroy_all_handlers(self):
        """销毁所有会话处理器"""
        logger.info("ConversationManager 销毁所有会话")

        async with self._lock:
            handlers = list(self.conversation_handlers.items())
            self.conversation_handlers.clear()

        for session_id, handler in handlers:
            await handler.stop()
