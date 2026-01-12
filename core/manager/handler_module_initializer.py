import asyncio
from typing import Dict, Optional, Callable

from adapters.protocols.protocol_factory import create_protocol_adapter
from modules.base_protocol import BaseProtocol
from utils.logging_setup import logger
from utils.module_initialization_utils import initialize_single_module_instance


class HandlerModuleInitializer:
    """
    一个工具类，用于初始化、管理和关闭通信模块（handlers）。
    不依赖于继承关系，并利用通用初始化工具。
    """

    def __init__(self, config: dict, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化处理器模块管理器。

        Args:
            config (dict): 全局配置字典。
            loop (Optional[asyncio.AbstractEventLoop]): asyncio 事件循环。
        """
        self.global_config = config
        self.event_loop = loop or asyncio.get_event_loop()
        self.modules: Dict[str, BaseProtocol] = {}  # 存储已加载的协议模块实例

        # 定义协议模块的创建工厂函数字典
        self.protocol_factories: Dict[str, Callable[..., BaseProtocol]] = {
            "protocols": create_protocol_adapter
        }
        logger.info("HandlerModuleInitializer 初始化完成，注册模块类型: %s", list(self.protocol_factories.keys()))

    async def initialize_modules(self):
        """
        根据全局配置初始化所有通信模块（handlers）。
        """
        logger.info("开始初始化所有通信模块...")
        module_configs = self.global_config.get("modules", {})

        if not module_configs:
            logger.warning("配置中未找到 'modules'，将不会初始化任何模块。")
            return

        protocol_config = module_configs.get("protocols")
        if protocol_config:
            # 对于 protocols 模块，我们假设只有一个顶层 "protocols" 配置
            await initialize_single_module_instance(
                module_id="protocols",
                module_config=protocol_config,
                factory_dict=self.protocol_factories,
                base_class=BaseProtocol,
                existing_modules=self.modules
            )
        else:
            logger.warning("配置中未找到 'protocols' 模块配置，将不会初始化任何协议。")

        logger.info("通信模块初始化完成，已加载模块: %s", list(self.modules.keys()))

    def get_module(self, module_id: str) -> Optional[BaseProtocol]:
        """获取指定 ID 的协议模块实例"""
        return self.modules.get(module_id)

    def get_all_modules(self) -> Dict[str, BaseProtocol]:
        """获取所有已加载的协议模块实例"""
        return self.modules

    async def shutdown_modules(self):
        """
        关闭所有已加载的通信模块。
        """
        logger.info("正在关闭所有通信模块...")
        for module_id, module in self.modules.items():
            module: BaseProtocol
            try:
                logger.info("关闭模块 '%s'...", module_id)
                await module.close()
                logger.info("模块 '%s' 关闭成功。", module_id)
            except Exception as e:
                logger.exception("关闭模块 '%s' 时出错: %s", module_id, e)

        self.modules.clear()
        logger.info("所有通信模块已关闭。")
