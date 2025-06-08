import asyncio
from typing import Dict, Optional, Callable

from adapters.handlers.handler_factory import create_input_handler
from modules.base_handler import BaseHandler
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
        self.modules: Dict[str, BaseHandler] = {}  # 存储已加载的处理器模块实例

        # 定义处理器模块的创建工厂函数字典
        self.handler_factories: Dict[str, Callable[..., BaseHandler]] = {
            "handlers": create_input_handler
        }
        logger.info("HandlerModuleInitializer 初始化完成，注册模块类型: %s", list(self.handler_factories.keys()))

    async def initialize_modules(self):
        """
        根据全局配置初始化所有通信模块（handlers）。
        """
        logger.info("开始初始化所有通信模块...")
        module_configs = self.global_config.get("modules", {})

        if not module_configs:
            logger.warning("配置中未找到 'modules'，将不会初始化任何模块。")
            return

        handler_config = module_configs.get("handlers")
        if handler_config:
            # 对于 handlers 模块，我们假设只有一个顶层 "handlers" 配置，
            # 里面包含各个 handler 的具体配置
            # 调用通用的初始化工具函数
            await initialize_single_module_instance(
                module_id="handlers",  # 这里的 ID 保持为 "handlers"
                module_config=handler_config,
                factory_dict=self.handler_factories,
                base_class=BaseHandler,
                existing_modules=self.modules
            )
        else:
            logger.warning("配置中未找到 'handlers' 模块配置，将不会初始化任何处理器。")

        logger.info("通信模块初始化完成，已加载模块: %s", list(self.modules.keys()))

    def get_module(self, module_id: str) -> Optional[BaseHandler]:
        """
        获取指定 ID 的处理器模块实例。

        Args:
            module_id (str): 模块的唯一标识符。

        Returns:
            Optional[BaseHandler]: 模块实例，如果不存在则返回 None。
        """
        return self.modules.get(module_id)

    def get_all_modules(self) -> Dict[str, BaseHandler]:
        """
        获取所有已加载的处理器模块实例。

        Returns:
            Dict[str, BaseHandler]: 包含所有模块实例的字典。
        """
        return self.modules

    async def shutdown_modules(self):
        """
        关闭所有已加载的通信模块。
        """
        logger.info("正在关闭所有通信模块...")
        for module_id, module in self.modules.items():
            module: BaseHandler
            try:
                logger.info("关闭模块 '%s'...", module_id)
                await module.close()
                logger.info("模块 '%s' 关闭成功。", module_id)
            except Exception as e:
                logger.exception("关闭模块 '%s' 时出错: %s", module_id, e)

        self.modules.clear()
        logger.info("所有通信模块已关闭。")
