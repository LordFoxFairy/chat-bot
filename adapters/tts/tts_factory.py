"""TTS (文本转语音) 适配器工厂模块"""

from typing import Any, Dict

from core.adapter_registry import AdapterRegistry, create_factory_function
from modules.base_tts import BaseTTS

# 创建 TTS 适配器注册器
tts_registry: AdapterRegistry[BaseTTS] = AdapterRegistry("TTS", BaseTTS)

# 注册可用的 TTS 适配器
tts_registry.register(
    "edge_tts",
    "adapters.tts.edge_tts_adapter"
)

# 向后兼容：导出工厂函数
create_tts_adapter = create_factory_function(tts_registry)

# 向后兼容：导出适配器加载器字典
TTS_ADAPTER_LOADERS = {
    adapter_type: lambda at=adapter_type: tts_registry._loaders[at]()
    for adapter_type in tts_registry.available_types
}
