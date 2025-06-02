from typing import Type, Optional, Dict, Any
import asyncio

# 导入 BaseTTS 用于类型提示
from modules.base_tts import BaseTTS #
from core.exceptions import ModuleInitializationError
from .edge_tts_adapter import EdgeTTSAdapter

TTS_ADAPTER_REGISTRY: Dict[str, Type[BaseTTS]] = {
    "edge_tts": EdgeTTSAdapter,
}

def create_tts_adapter(
    module_id: str,
    config: Optional[Dict[str, Any]] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    event_manager: Optional[Any] = None,
) -> BaseTTS:
    adapter_type = config.get("adapter_type")
    adapter_class = TTS_ADAPTER_REGISTRY.get(adapter_type)
    if not adapter_class:
        raise ModuleInitializationError(
            f"不支持的 TTS 适配器类型: '{adapter_type}'. "
            f"可用类型: {list(TTS_ADAPTER_REGISTRY.keys())}"
        )
    try:
        print(f"[TTS Factory] 正在为类型 '{adapter_type}' 创建实例 '{module_id}' 使用类 '{adapter_class.__name__}'")
        instance = adapter_class(
            module_id=module_id,
            config=config,
            event_loop=event_loop,
            event_manager=event_manager
        )
        return instance
    except Exception as e:
        raise ModuleInitializationError(
            f"创建 TTS 适配器 '{module_id}' (类型: '{adapter_type}') 失败: {e}"
        ) from e