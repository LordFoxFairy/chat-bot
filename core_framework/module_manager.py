import asyncio
import logging
from typing import Dict, Optional, Callable
from typing import TYPE_CHECKING

from adapters.asr.asr_factory import create_asr_adapter
from adapters.llm.llm_factory import create_llm_adapter
from adapters.tts.tts_factory import create_tts_adapter
from adapters.vad.vad_factory import create_vad_adapter
from core_framework.exceptions import ModuleInitializationError
from modules.base_module import BaseModule
from services.config_loader import ConfigLoader

if TYPE_CHECKING:
    from core_framework.event_manager import EventManager

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 实例


class ModuleManager:
    """
    模块管理器 (ModuleManager)
    负责加载、初始化、管理和卸载应用程序中的所有模块。
    它从配置文件中读取模块配置，并使用相应的创建函数（工厂函数）来实例化它们。
    """

    def __init__(self, config,
                 event_manager: Optional['EventManager'] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        初始化 ModuleManager。

        Args:
            config_loader (ConfigLoader): 配置加载器实例，用于加载全局应用配置。
                                          此实例应已配置好其主要的配置文件路径。
            event_manager (Optional[EventManager]): 应用的事件管理器实例，用于模块间的通信。
            loop (Optional[asyncio.AbstractEventLoop]): asyncio 事件循环。
                                                        如果为 None，则获取当前正在运行的事件循环。
        """
        self.global_config = config
        self.event_manager = event_manager
        self.event_loop = loop if loop else asyncio.get_event_loop()  # 获取或使用传入的事件循环

        self.modules: Dict[str, BaseModule] = {}  # 用于存储已加载模块的字典，键为 module_id，值为模块实例

        # 存储模块创建函数
        self.creation_functions: Dict[str, Callable[..., BaseModule]] = {
            "asr": create_asr_adapter,  # 语音识别模块的创建函数
            "llm": create_llm_adapter,  # 大语言模型模块的创建函数
            "tts": create_tts_adapter,  # 文本转语音模块的创建函数
            "vad": create_vad_adapter  # 语音活动检测模块的创建函数
        }
        logger.info("ModuleManager 初始化完成，已注册模块创建函数: %s", list(self.creation_functions.keys()))

    async def initialize_modules(self):
        """
        异步初始化所有模块。
        此方法会读取配置文件中 'module_configs' 部分定义的模块，
        使用相应的创建函数实例化它们，然后调用每个模块的异步 `initialize` 方法。
        """
        logger.info("开始初始化所有已配置的模块...")

        all_module_configs = self.global_config.get("modules", {})

        if not all_module_configs:
            logger.warning("在全局配置中未找到 'module_configs' 部分。"
                           "ModuleManager 将不会加载任何模块。")
            return

        for module_id, module_specific_config in all_module_configs.items():
            if not isinstance(module_specific_config, dict):
                logger.error(f"模块 ID '{module_id}' 的配置不是一个字典。跳过。")
                continue

            if module_id in self.modules:
                logger.warning(f"模块 ID '{module_id}' 已加载。跳过重新初始化。")
                continue

            creation_func = self.creation_functions.get(module_id)
            if not creation_func:
                logger.error(f"未找到模块类型 '{module_id}' (模块 ID '{module_id}' 需要) 的注册创建函数。跳过。")
                continue

            try:
                logger.info(f"尝试使用其创建函数创建模块 '{module_id}' (类型: '{module_id}')。")

                module_instance = creation_func(
                    module_id=module_id,
                    config=module_specific_config,
                    event_loop=self.event_loop,
                    event_manager=self.event_manager
                )

                if not module_instance:
                    raise ModuleInitializationError(
                        f"类型 '{module_id}' 的创建函数为模块 ID '{module_id}' 返回了 None。"
                        "创建函数应抛出错误或返回有效的模块实例。"
                    )

                if not isinstance(module_instance, BaseModule):
                    raise ModuleInitializationError(
                        f"由创建函数创建的模块 '{module_id}' (类型 '{module_id}') "
                        f"不是 BaseModule 的实例。实际类型: {type(module_instance).__name__}。"
                    )

                logger.info(
                    f"调用模块 '{module_id}' (类: {module_instance.__class__.__name__}) 的异步 initialize() 方法...")
                await module_instance.initialize()

                self.modules[module_id] = module_instance
                logger.info(f"模块 '{module_id}' (类型: {module_id}) 初始化并注册成功。")

            except ModuleInitializationError as e:
                logger.error(f"模块 '{module_id}' 发生 ModuleInitializationError: {e}")
            except Exception as e:
                logger.error(f"初始化模块 '{module_id}' (类型: {module_id}) 时发生意外错误: {e}", exc_info=True)

        logger.info(f"模块初始化过程完成。成功加载的模块: {list(self.modules.keys())}")

    def get_module(self, module_id: str) -> Optional[BaseModule]:
        """
        通过模块 ID 检索已加载的模块实例。
        """
        module = self.modules.get(module_id)
        if not module:
            logger.warning(f"尝试获取 ID 为 '{module_id}' 的模块，但未找到或未加载。")
        return module

    def get_all_modules(self) -> Dict[str, BaseModule]:
        """
        返回所有已加载模块实例的字典。
        """
        return self.modules

    async def shutdown_modules(self):
        """
        异步关闭所有已注册的模块。
        """
        logger.info("开始关闭所有已注册模块的序列...")
        for module_id, module_instance in self.modules.items():
            try:
                logger.info(f"尝试关闭模块 '{module_id}' (类: {module_instance.__class__.__name__})...")
                await module_instance.stop()
                logger.info(f"模块 '{module_id}' 关闭成功。")
            except Exception as e:
                logger.error(f"关闭模块 '{module_id}' 时发生错误: {e}", exc_info=True)

        self.modules.clear()
        logger.info("所有模块都已处理关闭。模块列表已清除。")
