"""VAD (语音活动检测) 适配器工厂模块"""

from src.core.adapter_registry import AdapterRegistry, create_factory_function
from src.core.interfaces.base_vad import BaseVAD

# 创建并配置注册器
vad_registry: AdapterRegistry[BaseVAD] = AdapterRegistry("VAD", BaseVAD)
vad_registry.register("silero_vad", "src.adapters.vad.silero_vad_adapter")

# 导出工厂函数
create_vad_adapter = create_factory_function(vad_registry)
