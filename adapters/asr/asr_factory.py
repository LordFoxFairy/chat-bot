from typing import Type, Optional, Dict, Any
import asyncio

from utils.logging_setup import logger
# 导入所有支持的 ASR 适配器类
from .funasr_sensevoice_adapter import FunASRSenseVoiceAdapter

# 导入 BaseASR 用于类型提示
from modules.base_asr import BaseASR  #
from core.exceptions import ModuleInitializationError

# 如果 EventManager 和 SessionManager 类型提示需要，可以导入
# from ...core_framework.event_manager import EventManager
# from ...core_framework.session_manager import SessionManager


# 映射 ASR 类型名称到对应的适配器类
ASR_ADAPTER_REGISTRY: Dict[str, Type[BaseASR]] = {
    "funasr_sensevoice": FunASRSenseVoiceAdapter,
    # 未来可以添加更多 ASR 类型，例如:
    # "whisper_local": WhisperLocalAdapter,
    # "azure_speech": AzureSpeechASRAdapter,
}


def create_asr_adapter(
        module_id: str,
        config: Optional[Dict[str, Any]] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> BaseASR:
    """
    根据指定的 adapter_type 创建并返回一个 ASR 适配器实例。

    参数:
        module_id (str): 要分配给模块实例的 ID.
        config (Optional[Dict[str, Any]]): 模块的配置字典.
        event_loop (Optional[asyncio.AbstractEventLoop]): 事件循环.

    返回:
        BaseASR: 一个 ASR 适配器的实例.

    抛出:
        ModuleInitializationError: 如果指定的 adapter_type 不被支持或实例化失败.
    """
    adapter_type = config.get("adapter_type")
    adapter_class = ASR_ADAPTER_REGISTRY.get(adapter_type)
    if not adapter_class:
        raise ModuleInitializationError(
            f"不支持的 ASR 适配器类型: '{adapter_type}'. "
            f"可用类型: {list(ASR_ADAPTER_REGISTRY.keys())}"
        )

    try:
        logger.info(
            f"[ASR Factory] 正在为类型 '{adapter_type}' 创建实例 '{module_id}' 使用类 '{adapter_class.__name__}'")
        # 注意：这里不传递 session_manager，因为 BaseASR 的构造函数不接收它
        # 如果将来有 ASR 适配器需要，可以调整
        instance = adapter_class(
            module_id=module_id,
            config=config,
            event_loop=event_loop
        )
        return instance
    except Exception as e:
        raise ModuleInitializationError(
            f"创建 ASR 适配器 '{module_id}' (类型: '{adapter_type}') 失败: {e}"
        ) from e
