from typing import Dict, Type, Any

from core.exceptions import ModuleInitializationError
from modules.base_tts import BaseTTS
from utils.logging_setup import logger

from .edge_tts_adapter import EdgeTTSAdapter


# TTS 适配器注册表
TTS_ADAPTERS: Dict[str, Type[BaseTTS]] = {
    "edge_tts": EdgeTTSAdapter,
    # 未来可以添加更多 TTS 适配器:
    # "azure_speech": AzureSpeechAdapter,
    # "google_tts": GoogleTTSAdapter,
}


def create_tts_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
) -> BaseTTS:
    """创建 TTS 适配器实例"""
    adapter_class = TTS_ADAPTERS.get(adapter_type)

    if adapter_class is None:
        available_types = list(TTS_ADAPTERS.keys())
        raise ModuleInitializationError(
            f"不支持的 TTS 适配器类型: '{adapter_type}'. "
            f"可用类型: {available_types}"
        )

    try:
        logger.info(
            f"TTS Factory: 创建 '{adapter_type}' 适配器，"
            f"模块ID: {module_id}，类: {adapter_class.__name__}"
        )

        instance = adapter_class(
            module_id=module_id,
            config=config,
        )

        return instance

    except Exception as e:
        raise ModuleInitializationError(
            f"创建 TTS 适配器 '{adapter_type}' 失败: {e}"
        ) from e
