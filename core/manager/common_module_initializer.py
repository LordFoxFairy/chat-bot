import asyncio
from typing import Dict, Optional, Callable, TYPE_CHECKING

# 导入必要的模块工厂函数和基类
from adapters.asr.asr_factory import create_asr_adapter
from adapters.llm.llm_factory import create_llm_adapter
from adapters.tts.tts_factory import create_tts_adapter
from adapters.vad.vad_factory import create_vad_adapter
from modules.base_module import BaseModule
from utils.logging_setup import logger
from utils.module_initialization_utils import initialize_single_module_instance

if TYPE_CHECKING:
    pass


class CommonModuleInitializer:
    """
    一个工具类，用于初始化、管理和关闭通用模块（如ASR, LLM, TTS, VAD）。
    不依赖于继承关系，并利用通用初始化工具。
    """

    def __init__(self, config: dict,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化通用模块管理器。

        Args:
            config (dict): 全局配置字典。
            loop (Optional[asyncio.AbstractEventLoop]): asyncio 事件循环。
        """
        self.global_config = config
        self.event_loop = loop or asyncio.get_event_loop()
        self.modules: Dict[str, BaseModule] = {}  # 存储已加载的模块实例

        # 定义通用模块的创建工厂函数字典
        self.module_factories: Dict[str, Callable[..., BaseModule]] = {
            "asr": create_asr_adapter,
            "llm": create_llm_adapter,
            "tts": create_tts_adapter,
            "vad": create_vad_adapter
        }
        logger.info("CommonModuleInitializer 初始化完成，注册模块类型: %s", list(self.module_factories.keys()))

    async def initialize_modules(self):
        """
        根据全局配置初始化所有通用模块。
        """
        logger.info("开始初始化所有通用模块...")
        module_configs = self.global_config.get("modules", {})

        if not module_configs:
            logger.warning("配置中未找到 'modules'，将不会初始化任何模块。")
            return

        # 遍历配置并初始化每个模块
        for module_id, module_config in module_configs.items():
            # 排除 handlers，因为它们由 HandlerModuleInitializer 处理
            if module_id == "handlers":
                continue
            else:
                # 调用通用的初始化工具函数
                await initialize_single_module_instance(
                    module_id=module_id,
                    module_config=module_config,
                    factory_dict=self.module_factories,
                    base_class=BaseModule,
                    existing_modules=self.modules,
                )

        logger.info("通用模块初始化完成，已加载模块: %s", list(self.modules.keys()))

    def get_module(self, module_id: str) -> Optional[BaseModule]:
        """
        获取指定 ID 的模块实例。

        Args:
            module_id (str): 模块的唯一标识符。

        Returns:
            Optional[BaseModule]: 模块实例，如果不存在则返回 None。
        """
        return self.modules.get(module_id)

    def get_all_modules(self) -> Dict[str, BaseModule]:
        """
        获取所有已加载的模块实例。

        Returns:
            Dict[str, BaseModule]: 包含所有模块实例的字典。
        """
        return self.modules

    async def shutdown_modules(self):
        """
        关闭所有已加载的通用模块。
        """
        logger.info("正在关闭所有通用模块...")
        for module_id, module in self.modules.items():
            try:
                logger.info("关闭模块 '%s'...", module_id)
                await module.close()
                logger.info("模块 '%s' 关闭成功。", module_id)
            except Exception as e:
                logger.exception("关闭模块 '%s' 时出错: %s", module_id, e)

        self.modules.clear()
        logger.info("所有通用模块已关闭。")
