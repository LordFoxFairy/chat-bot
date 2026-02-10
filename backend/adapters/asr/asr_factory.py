"""ASR (语音识别) 适配器工厂模块"""

from backend.core.adapter_registry import AdapterRegistry, create_factory_function
from backend.core.interfaces.base_asr import BaseASR

# 创建并配置注册器
asr_registry: AdapterRegistry[BaseASR] = AdapterRegistry("ASR", BaseASR)
asr_registry.register("funasr_sensevoice", "src.adapters.asr.funasr_sensevoice_adapter")

# 导出工厂函数
create_asr_adapter = create_factory_function(asr_registry)
