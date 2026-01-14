from typing import Dict, Optional, Callable, Awaitable

from models import StreamEvent
from utils.logging_setup import logger
from .conversation import ConversationHandler
from .session_manager import SessionManager


class ConversationManager:
    """会话管理器

    职责:
    - 管理所有 ConversationHandler 的生命周期
    - 提供会话创建、获取、销毁接口
    - 不涉及协议传输（由 Protocol 负责）
    - 不涉及模块管理（由 ChatEngine 负责）
    """

    def __init__(self, chat_engine: 'ChatEngine', session_manager: SessionManager):
        self.chat_engine = chat_engine
        self.session_manager = session_manager
        self.conversation_handlers: Dict[str, ConversationHandler] = {}
        logger.info("ConversationManager 初始化完成")

    async def create_conversation_handler(
        self,
        session_id: str,
        tag_id: str,
        send_callback: Callable[[StreamEvent], Awaitable[None]]
    ) -> ConversationHandler:
        """创建并启动 ConversationHandler

        Args:
            session_id: 会话唯一标识
            tag_id: 客户端标签
            send_callback: 消息发送回调函数

        Returns:
            ConversationHandler: 创建的会话处理器
        """
        if session_id in self.conversation_handlers:
            logger.warning(f"ConversationManager 会话已存在: {session_id}")
            return self.conversation_handlers[session_id]

        handler = ConversationHandler(
            session_id=session_id,
            tag_id=tag_id,
            chat_engine=self.chat_engine,
            session_manager=self.session_manager,
            send_callback=send_callback
        )
        await handler.start()

        self.conversation_handlers[session_id] = handler
        logger.info(f"ConversationManager 创建会话: session={session_id}, tag={tag_id}")

        return handler

    def get_conversation_handler(self, session_id: str) -> Optional[ConversationHandler]:
        """获取会话处理器

        Args:
            session_id: 会话唯一标识

        Returns:
            Optional[ConversationHandler]: 会话处理器，不存在则返回 None
        """
        return self.conversation_handlers.get(session_id)

    async def destroy_conversation_handler(self, session_id: str):
        """销毁会话处理器

        Args:
            session_id: 会话唯一标识
        """
        handler = self.conversation_handlers.pop(session_id, None)
        if handler:
            await handler.stop()
            logger.info(f"ConversationManager 销毁会话: session={session_id}")
        else:
            logger.warning(f"ConversationManager 会话不存在: {session_id}")

    async def destroy_all_handlers(self):
        """销毁所有会话处理器"""
        logger.info("ConversationManager 销毁所有会话")

        for session_id, handler in list(self.conversation_handlers.items()):
            await handler.stop()

        self.conversation_handlers.clear()
