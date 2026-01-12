from typing import Dict, Type, Any

from core.exceptions import ModuleInitializationError
from modules.base_vad import BaseVAD
from utils.logging_setup import logger

from .silero_vad_adapter import SileroVADAdapter


# VAD 适配器注册表
VAD_ADAPTERS: Dict[str, Type[BaseVAD]] = {
    "silero_vad": SileroVADAdapter,
    # 未来可以添加更多 VAD 适配器:
    # "webrtc_vad": WebRTCVADAdapter,
}


def create_vad_adapter(
    adapter_type: str,
    module_id: str,
    config: Dict[str, Any],
) -> BaseVAD:
    """创建 VAD 适配器实例"""
    adapter_class = VAD_ADAPTERS.get(adapter_type)

    if adapter_class is None:
        available_types = list(VAD_ADAPTERS.keys())
        raise ModuleInitializationError(
            f"不支持的 VAD 适配器类型: '{adapter_type}'. "
            f"可用类型: {available_types}"
        )

    try:
        logger.info(
            f"VAD Factory: 创建 '{adapter_type}' 适配器，"
            f"模块ID: {module_id}，类: {adapter_class.__name__}"
        )

        instance = adapter_class(
            module_id=module_id,
            config=config,
        )

        return instance

    except Exception as e:
        raise ModuleInitializationError(
            f"创建 VAD 适配器 '{adapter_type}' 失败: {e}"
        ) from e
