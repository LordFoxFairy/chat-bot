from abc import abstractmethod
from typing import AsyncGenerator, Dict, Any

from backend.core.models import AudioData, TextData
from backend.core.interfaces.base_module import BaseModule
from backend.utils.logging_setup import logger


class BaseTTS(BaseModule):
    """文本转语音模块基类

    职责:
    - 定义 TTS 核心接口
    - 提供通用的文本处理流程

    子类需要实现:
    - synthesize_stream: 流式合成语音
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取 TTS 通用配置
        self.voice = self.config.get("voice", "zh-CN-XiaoxiaoNeural")
        self.sample_rate = self.config.get("sample_rate", 24000)
        self.channels = self.config.get("channels", 1)

        logger.debug(f"TTS [{self.module_id}] 配置加载:")
        logger.debug(f"  - voice: {self.voice}")
        logger.debug(f"  - sample_rate: {self.sample_rate}")
        logger.debug(f"  - channels: {self.channels}")

    @abstractmethod
    async def synthesize_stream(self, text: TextData) -> AsyncGenerator[AudioData, None]:
        """流式合成语音，返回音频流"""
        raise NotImplementedError("TTS 子类必须实现 synthesize_stream 方法")

    async def _setup_impl(self):
        """初始化逻辑（默认为空，子类可覆盖）"""
        pass

    async def process_text(
        self,
        text: TextData,
        session_id: str
    ) -> AsyncGenerator[AudioData, None]:
        """处理文本并生成音频流，内部调用 synthesize_stream"""
        if not self.is_ready:
            from backend.core.models.exceptions import ModuleProcessingError
            raise ModuleProcessingError(f"TTS 模块 {self.module_id} 未就绪")

        try:
            logger.info(f"TTS [{self.module_id}] (Session: {session_id}) 开始合成: '{text.text[:30]}...'")

            async for audio_chunk in self.synthesize_stream(text):
                yield audio_chunk

            logger.info(f"TTS [{self.module_id}] (Session: {session_id}) 合成完成")

        except Exception as e:
            logger.error(
                f"TTS [{self.module_id}] (Session: {session_id}) 合成失败: {e}",
                exc_info=True
            )
            from backend.core.models import AudioFormat
            yield AudioData(
                data=b"",
                format=AudioFormat.MP3,
                is_final=True,
                metadata={
                    "error": str(e),
                    "source_module_id": self.module_id
                }
            )
