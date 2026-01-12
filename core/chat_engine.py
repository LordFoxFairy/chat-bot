import asyncio
from typing import Dict, Any, Optional

import constant
from utils.logging_setup import logger
from .manager.common_module_initializer import CommonModuleInitializer
from .manager.handler_module_initializer import HandlerModuleInitializer
from .session_context import SessionContext
from .session_manager import session_manager
from .conversation_manager import ConversationManager


class ChatEngine:
    """聊天引擎

    职责:
    - 加载和管理所有模块 (ASR, LLM, TTS, VAD, Protocols)
    - 提供模块访问接口
    - 提供 ConversationManager 实例（不直接管理会话）
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

        # 会话管理器
        self.conversation_manager = ConversationManager(chat_engine=self)

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

            # 初始化 Handler 模块 (Protocols) - 传递 ConversationManager
            await self.handler_manager.initialize_modules(
                conversation_manager=self.conversation_manager
            )

            logger.info("ChatEngine 模块初始化完成")

        except Exception as e:
            logger.critical(f"ChatEngine 模块加载失败: {e}", exc_info=True)
            raise

    def get_module(self, module_name: str):
        """获取模块实例"""
        return self.module_manager.get_module(module_name)
