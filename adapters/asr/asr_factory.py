from typing import Dict, Type, Any

from core.exceptions import ModuleInitializationError
from modules.base_asr import BaseASR
from utils.logging_setup import logger

from .funasr_sensevoice_adapter import FunASRSenseVoiceAdapter


# ASR 适配器注册表
ASR_ADAPTERS: Dict[str, Type[BaseASR]] = {
    "funasr_sensevoice": FunASRSenseVoiceAdapter,
    # 未来可以添加更多 ASR 适配器:
    # "whisper": WhisperAdapter,
    # "azure_speech": AzureSpeechAdapter,
}


def create_asr_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
) -> BaseASR:
    """创建 ASR 适配器实例"""
    adapter_class = ASR_ADAPTERS.get(adapter_type)

    if adapter_class is None:
        available_types = list(ASR_ADAPTERS.keys())
        raise ModuleInitializationError(
            f"不支持的 ASR 适配器类型: '{adapter_type}'. "
            f"可用类型: {available_types}"
        )

    try:
        logger.info(
            f"ASR Factory: 创建 '{adapter_type}' 适配器，"
            f"模块ID: {module_id}，类: {adapter_class.__name__}"
        )

        instance = adapter_class(
            module_id=module_id,
            config=config,
        )

        return instance

    except Exception as e:
        raise ModuleInitializationError(
            f"创建 ASR 适配器 '{adapter_type}' 失败: {e}"
        ) from e
