"""ASR (语音识别) 适配器工厂模块"""

from typing import Any, Dict

from src.core.adapter_registry import AdapterRegistry, create_factory_function
from src.core.interfaces.base_asr import BaseASR

# 创建 ASR 适配器注册器
asr_registry: AdapterRegistry[BaseASR] = AdapterRegistry("ASR", BaseASR)

# 注册可用的 ASR 适配器
asr_registry.register(
    "funasr_sensevoice",
    "src.adapters.asr.funasr_sensevoice_adapter"
)

# 向后兼容：导出工厂函数
create_asr_adapter = create_factory_function(asr_registry)

# 向后兼容：导出适配器加载器字典
ASR_ADAPTER_LOADERS = {
    adapter_type: lambda at=adapter_type: asr_registry._loaders[at]()
    for adapter_type in asr_registry.available_types
}
