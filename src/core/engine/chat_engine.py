"""聊天引擎

ChatEngine 是应用的核心协调器，负责：
- 初始化和管理所有模块
- 提供模块访问接口
- 管理应用生命周期
"""

from typing import Any, Dict, Optional, TYPE_CHECKING

from src.core.interfaces.base_module import BaseModule
from src.core.interfaces.base_protocol import BaseProtocol
from src.core.session.session_manager import SessionManager
from src.utils.logging_setup import logger
from src.utils.module_initialization_utils import initialize_single_module_instance

if TYPE_CHECKING:
    from src.core.adapter_loader import AdapterLoader


class ChatEngine:
    """聊天引擎

    职责:
    - 加载和管理所有模块 (ASR, LLM, TTS, VAD)
    - 提供模块访问接口
    - 管理应用生命周期
    """

    def __init__(
        self,
        config: Dict[str, Any],
        session_manager: SessionManager,
        adapter_loader: Optional["AdapterLoader"] = None,
    ):
        """初始化聊天引擎

        Args:
            config: 全局配置
            session_manager: 会话管理器
            adapter_loader: 适配器加载器（可选，默认创建内置加载器）
        """
        self.global_config = config
        self.session_manager = session_manager

        # 适配器加载器（延迟导入避免循环依赖）
        if adapter_loader is None:
            from src.core.adapter_loader import create_default_loader
            adapter_loader = create_default_loader()
        self.adapter_loader = adapter_loader

        # 模块存储
        self.common_modules: Dict[str, BaseModule] = {}
        self.protocol_modules: Dict[str, BaseProtocol] = {}

        # 会话管理器
        from src.core.session.conversation_manager import ConversationManager
        self.conversation_manager = ConversationManager(session_manager=session_manager)

        logger.info("ChatEngine 初始化完成")

    async def initialize(self) -> None:
        """异步初始化 ChatEngine，加载所有模块"""
        logger.info("ChatEngine 正在初始化模块...")

        try:
            # 构建工厂字典
            common_factories = {
                module_type: lambda adapter_type, module_id, config, mt=module_type, **kw: self.adapter_loader.create(
                    mt, adapter_type, module_id, config, **kw
                )
                for module_type in ["asr", "llm", "tts", "vad"]
                if self.adapter_loader.has_factory(module_type)
            }

            # 获取模块配置
            module_configs = self.global_config.get("modules", {})
            if not module_configs:
                logger.warning("配置中未找到 'modules'，将不会初始化任何模块")
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
                        raise_on_error=True  # 在初始化阶段，任何致命错误都应该抛出异常
                    )

            # 初始化协议模块
            protocol_config = module_configs.get("protocols")
            if protocol_config and self.adapter_loader.has_factory("protocol"):
                protocol_factory = {
                    "protocols": lambda adapter_type, module_id, config, **kw: self.adapter_loader.create(
                        "protocol", adapter_type, module_id, config, **kw
                    )
                }
                await initialize_single_module_instance(
                    module_id="protocols",
                    module_config=protocol_config,
                    factory_dict=protocol_factory,
                    base_class=BaseProtocol,
                    existing_modules=self.protocol_modules,
                    conversation_manager=self.conversation_manager,
                    raise_on_error=True  # 在初始化阶段，任何致命错误都应该抛出异常
                )

            # 设置全局模块上下文
            from src.core.app_context import AppContext
            AppContext.set_modules(self.common_modules)

            logger.info("ChatEngine 模块初始化完成")

        except Exception as e:
            logger.critical(f"ChatEngine 模块加载失败: {e}", exc_info=True)
            raise

    def get_module(self, module_name: str) -> Optional[BaseModule]:
        """获取模块实例"""
        return self.common_modules.get(module_name)

    async def shutdown(self) -> None:
        """关闭引擎，释放所有资源"""
        logger.info("ChatEngine 正在关闭...")

        # 关闭所有通用模块
        for module_id, module in self.common_modules.items():
            try:
                await module.close()
                logger.debug(f"模块 {module_id} 已关闭")
            except Exception as e:
                logger.error(f"关闭模块 {module_id} 失败: {e}", exc_info=True)

        # 关闭所有协议模块
        for protocol_id, protocol in self.protocol_modules.items():
            try:
                await protocol.close()
                logger.debug(f"协议 {protocol_id} 已关闭")
            except Exception as e:
                logger.error(f"关闭协议 {protocol_id} 失败: {e}", exc_info=True)

        self.common_modules.clear()
        self.protocol_modules.clear()

        logger.info("ChatEngine 已关闭")

    async def health_check(self) -> Dict[str, Any]:
        """系统健康检查

        Returns:
            各模块健康状态
        """
        result = {
            "engine": "healthy",
            "modules": {},
        }

        for module_id, module in self.common_modules.items():
            try:
                result["modules"][module_id] = await module.health_check()
            except Exception as e:
                result["modules"][module_id] = {
                    "status": "error",
                    "error": str(e),
                }

        return result
