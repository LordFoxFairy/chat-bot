import asyncio
import logging
from abc import abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any
from data_models.audio_data import AudioData
from modules.base_module import BaseModule
from core_framework.event_manager import EventManager

logger = logging.getLogger(__name__)


class BaseVAD(BaseModule):
    """
    语音活动检测 (VAD) 模块的抽象基类。
    """

    def __init__(self, module_id: str,
                 module_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 event_manager: Optional['EventManager'] = None):

        # 3. 调用父类 BaseModule 的 __init__ 方法
        super().__init__(
            module_id=module_id,
            config=config,  # 传递提取的模块特定配置
            event_manager=event_manager,
            event_loop=event_loop
        )

        self.vad_processing_config = self.config.get('vad', {})

        self.default_sample_rate = self.vad_processing_config.get('default_sample_rate', 16000)

        logger.info(f"BaseVAD 模块 (id='{self.module_id}') 初始化完成。 "
                         f"vad_config 中的默认采样率: {self.default_sample_rate} Hz。")

    @abstractmethod
    def get_config_key(self) -> str:
        """
        返回用于在 config.yaml 的 'vad_config' 部分中
        访问此 VAD 适配器特定配置的键。

        示例: "silero_vad"
        """
        pass

    @abstractmethod
    async def process(self, audio_data: AudioData) -> AsyncGenerator[AudioData, None]:
        """
        处理传入的音频块以进行语音活动检测。

        此方法应该是一个异步生成器，产生 `AudioData` 对象。
        每个产生的 `AudioData` 对象应代表一个检测到的语音片段。
        此生成器产生的 *输出* `AudioData` 对象上的 `is_final` 标志应为 `True`，
        以指示这是 VAD 检测到的一个完整语音片段。

        *输入* `audio_data` 上的 `is_final` 标志指示它是否是来自客户端的
        整体输入音频流的最后一个块。VAD 实现应适当处理此信号，
        例如，通过刷新任何内部缓冲区并重置其状态。

        Args:
            audio_data (AudioData): 包含要处理的音频块的 AudioData 对象。

        Yields:
            AudioData: 代表检测到的语音片段的 AudioData 对象。
                       此产生的对象上的 `is_final` 标志应为 True。
        """
        # 这是一个抽象方法，因此它需要是一个异步生成器。
        # `if False: yield` 结构确保 linter/类型检查器将其识别为此类。
        if False:  # pragma: no cover
            yield

    async def reset_state(self):
        """
        重置 VAD 模块的任何内部状态。
        这通常在新的音频流开始或会话结束时调用，
        确保 VAD 从新的状态开始，而不携带先前处理的状态。
        子类应实现此方法以清除缓冲区、重置模型等。
        """
        logger.info(f"为 {self.get_config_key()} 调用了 BaseVAD 状态重置。")
        pass  # 默认实现不执行任何操作。

    async def close(self):
        """
        异步关闭或释放 VAD 模块持有的任何资源。
        这可能包括从内存中卸载模型、关闭文件句柄等。
        如果子类管理此类资源，则应实现此方法。
        """
        logger.info(f"为 {self.get_config_key()} 调用了 BaseVAD 关闭。")
        pass  # 默认实现不执行任何操作。

