import asyncio
from typing import Dict, Any, Optional

import constant
from utils.logging_setup import logger
from .manager.common_module_initializer import CommonModuleInitializer
from .manager.handler_module_initializer import HandlerModuleInitializer
from .session_context import SessionContext
from .session_manager import session_manager


class ChatEngine:
    """
    聊天引擎，负责加载模块和监听WebSocket连接，打印接收到的数据。
    """

    def __init__(self,
                 config: Dict[str, Any],
                 loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_event_loop()):
        self.global_config = config
        self.loop = loop if loop else asyncio.get_event_loop()
        self.module_manager = CommonModuleInitializer(config=self.global_config)
        self.handler_manager = HandlerModuleInitializer(config=self.global_config)
        self.active_sessions: Dict[str, SessionContext] = {}
        self.handler = None
        logger.info("全局 ChatEngine 初始化完成。")

    async def initialize(self):
        """
        异步初始化 ChatEngine，仅加载模块。
        """
        logger.info("ChatEngine 正在进行异步初始化 (初始化模块)...")
        try:
            await self.module_manager.initialize_modules()
            context = SessionContext()
            context.global_module_manager = self.module_manager
            context.tag_id = constant.CHAT_ENGINE_NAME
            context.session_id = constant.CHAT_ENGINE_NAME
            session_manager.create_session(context)
            await self.handler_manager.initialize_modules()
            logger.info("ChatEngine 的 ModuleManager 初始化完成。")
        except Exception as e:
            logger.critical(f"ChatEngine: 模块加载失败: {e}", exc_info=True)
            raise  # 如果模块加载失败，则抛出异常停止服务器
