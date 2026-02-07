from abc import abstractmethod
from typing import Dict, Any

from modules.base_module import BaseModule
from utils.logging_setup import logger


class BaseVAD(BaseModule):
    """语音活动检测模块基类

    职责:
    - 定义 VAD 核心接口
    - 提供通用的语音检测流程

    子类需要实现:
    - detect: 检测音频中是否包含语音
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取 VAD 通用配置
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.threshold = self.config.get("threshold", 0.5)

        logger.debug(f"VAD [{self.module_id}] 配置加载:")
        logger.debug(f"  - sample_rate: {self.sample_rate}")
        logger.debug(f"  - threshold: {self.threshold}")

    @abstractmethod
    async def detect(self, audio_data: bytes) -> bool:
        """检测音频中是否包含语音"""
        raise NotImplementedError("VAD 子类必须实现 detect 方法")

    async def _setup_impl(self):
        """初始化逻辑（默认为空，子类可覆盖）"""
        pass

    async def reset_state(self):
        """重置 VAD 内部状态"""
        logger.debug(f"VAD [{self.module_id}] 重置状态")
