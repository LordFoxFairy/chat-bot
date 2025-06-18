import asyncio
from utils.logging_setup import logger
from abc import abstractmethod
from typing import Optional, Dict, Any

from data_models.audio_data import AudioData
from modules.base_module import BaseModule




class BaseVAD(BaseModule):
    """
    语音活动检测 (VAD) 模块的抽象基类。
    新接口: is_speech_present 用于判断单个音频窗口是否包含语音。
    """

    def __init__(self, module_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None):
        super().__init__(
            module_id=module_id,
            config=config,
            event_loop=event_loop
        )
        self.default_sample_rate: int = int(self.config.get('default_sample_rate', 16000))
        logger.info(f"BaseVAD 模块 (id='{self.module_id}') 初始化完成。 "
                    f"默认采样率: {self.default_sample_rate} Hz。")

    @abstractmethod
    def get_config_key(self) -> str:
        """
        返回用于在 config.yaml 的 'modules.vad.config' 部分中
        访问此 VAD 适配器特定配置的键。
        例如: "silero_vad"
        """
        pass

    @abstractmethod
    async def is_speech_present(self, audio_data: bytes) -> bool:
        """
        判断传入的单个音频窗口是否包含语音。
        ChatEngine 应确保传入的 audio_data 包含一个适合VAD处理的音频窗口。

        Args:
            audio_data : 包含单个音频窗口及其元数据的对象。

        Returns:
            bool: True 如果当前窗口有语音，否则 False。
                  如果发生严重错误无法判断，也应返回 False 并记录错误。
        """
        raise NotImplementedError("VAD 子类必须实现 is_speech_present 方法。")

    async def reset_state(self):
        """
        重置 VAD 模块的任何内部状态。
        """
        logger.info(f"BaseVAD [{self.module_id}] reset_state called.")
        pass

    async def close(self):
        """
        异步关闭或释放 VAD 模块持有的任何资源。
        """
        logger.info(f"BaseVAD [{self.module_id}] close called.")
        pass
