import asyncio
from typing import Dict, Any, Optional

import constant
from utils.logging_setup import logger
from utils.module_initialization_utils import initialize_single_module_instance
from .session_context import SessionContext
from .session_manager import session_manager
from .conversation_manager import ConversationManager

# 导入模块工厂函数
from adapters.asr.asr_factory import create_asr_adapter
from adapters.llm.llm_factory import create_llm_adapter
from adapters.tts.tts_factory import create_tts_adapter
from adapters.vad.vad_factory import create_vad_adapter
from adapters.protocols.protocol_factory import create_protocol_adapter

from modules.base_module import BaseModule
from modules.base_protocol import BaseProtocol


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

        # 模块存储
        self.common_modules: Dict[str, BaseModule] = {}  # ASR, LLM, TTS, VAD
        self.protocol_modules: Dict[str, BaseProtocol] = {}  # Protocols

        # 会话管理器
        self.conversation_manager = ConversationManager(chat_engine=self)

        logger.info("ChatEngine 初始化完成")

    async def initialize(self):
        """异步初始化 ChatEngine，加载所有模块"""
        logger.info("ChatEngine 正在初始化模块...")

        try:
            # 定义模块工厂
            common_factories = {
                "asr": create_asr_adapter,
                "llm": create_llm_adapter,
                "tts": create_tts_adapter,
                "vad": create_vad_adapter,
            }

            # 获取模块配置
            module_configs = self.global_config.get("modules", {})
            if not module_configs:
                logger.warning("配置中未找到 'modules'，将不会初始化任何模块。")
                return

            # 初始化通用模块 (ASR, LLM, TTS, VAD)
            for module_id, module_config in module_configs.items():
                if module_id in common_factories:
                    await initialize_single_module_instance(
                        module_id=module_id,
                        module_config=module_config,
                        factory_dict=common_factories,
                        base_class=BaseModule,
                        existing_modules=self.common_modules,
                    )

            # 初始化协议模块 (Protocols)
            protocol_config = module_configs.get("protocols")
            if protocol_config:
                await initialize_single_module_instance(
                    module_id="protocols",
                    module_config=protocol_config,
                    factory_dict={"protocols": create_protocol_adapter},
                    base_class=BaseProtocol,
                    existing_modules=self.protocol_modules,
                    conversation_manager=self.conversation_manager
                )

            logger.info("ChatEngine 模块初始化完成")

        except Exception as e:
            logger.critical(f"ChatEngine 模块加载失败: {e}", exc_info=True)
            raise

    def get_module(self, module_name: str):
        """获取模块实例"""
        return self.common_modules.get(module_name)
