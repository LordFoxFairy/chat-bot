"""TTS (文本转语音) 适配器工厂模块"""

from backend.core.adapter_registry import AdapterRegistry, create_factory_function
from backend.core.interfaces.base_tts import BaseTTS

# 创建并配置注册器
tts_registry: AdapterRegistry[BaseTTS] = AdapterRegistry("TTS", BaseTTS)
tts_registry.register("edge_tts", "src.adapters.tts.edge_tts_adapter")

# 导出工厂函数
create_tts_adapter = create_factory_function(tts_registry)
