import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable

import constant
from data_models import StreamEvent
from utils.logging_setup import logger
from .manager.common_module_initializer import CommonModuleInitializer
from .manager.handler_module_initializer import HandlerModuleInitializer
from .session_context import SessionContext
from .session_manager import session_manager
from .conversation_handler import ConversationHandler


class ChatEngine:
    """聊天引擎

    职责:
    - 加载和管理所有模块 (ASR, LLM, TTS, VAD, Protocols)
    - 管理所有 ConversationHandler 的生命周期
    - 提供模块访问接口
    """

    def __init__(
        self,
        config: Dict[str, Any],
        loop: Optional[asyncio.AbstractEventLoop] = None
    ):
        self.global_config = config
        self.loop = loop if loop else asyncio.get_event_loop()

        # 模块管理器
        self.module_manager = CommonModuleInitializer(config=self.global_config)
        self.handler_manager = HandlerModuleInitializer(config=self.global_config)

        # ConversationHandler 管理
        self.conversation_handlers: Dict[str, ConversationHandler] = {}

        logger.info("ChatEngine 初始化完成")

    async def initialize(self):
        """异步初始化 ChatEngine，加载所有模块"""
        logger.info("ChatEngine 正在初始化模块...")

        try:
            # 初始化通用模块 (ASR, LLM, TTS, VAD)
            await self.module_manager.initialize_modules()

            # 创建全局 SessionContext
            context = SessionContext()
            context.global_module_manager = self.module_manager
            context.tag_id = constant.CHAT_ENGINE_NAME
            context.session_id = constant.CHAT_ENGINE_NAME
            session_manager.create_session(context)

            # 初始化 Handler 模块 (Protocols) - 传递自己
            await self.handler_manager.initialize_modules(chat_engine=self)

            logger.info("ChatEngine 模块初始化完成")

        except Exception as e:
            logger.critical(f"ChatEngine 模块加载失败: {e}", exc_info=True)
            raise

    def get_module(self, module_name: str):
        """获取模块实例"""
        return self.module_manager.get_module(module_name)

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
            logger.warning(f"ChatEngine 会话已存在: {session_id}")
            return self.conversation_handlers[session_id]

        handler = ConversationHandler(
            session_id=session_id,
            tag_id=tag_id,
            chat_engine=self,
            send_callback=send_callback
        )
        await handler.start()

        self.conversation_handlers[session_id] = handler
        logger.info(f"ChatEngine 创建会话处理器: session={session_id}, tag={tag_id}")

        return handler

    def get_conversation_handler(self, session_id: str) -> Optional[ConversationHandler]:
        """获取会话处理器"""
        return self.conversation_handlers.get(session_id)

    async def destroy_conversation_handler(self, session_id: str):
        """销毁会话处理器

        Args:
            session_id: 会话唯一标识
        """
        handler = self.conversation_handlers.pop(session_id, None)
        if handler:
            await handler.stop()
            logger.info(f"ChatEngine 销毁会话处理器: session={session_id}")
        else:
            logger.warning(f"ChatEngine 会话不存在: {session_id}")
