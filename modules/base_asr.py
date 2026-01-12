from abc import abstractmethod
from typing import Dict, Any, AsyncGenerator

from core.exceptions import ModuleProcessingError
from data_models import AudioData, TextData
from modules.base_module import BaseModule
from utils.logging_setup import logger


class BaseASR(BaseModule):
    """语音识别模块基类

    职责:
    - 定义 ASR 核心接口
    - 提供通用的音频处理流程

    子类需要实现:
    - recognize: 识别单个音频
    """

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        # 读取 ASR 通用配置
        self.language = self.config.get("language", "zh-CN")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.channels = self.config.get("channels", 1)

        logger.debug(f"ASR [{self.module_id}] 配置加载:")
        logger.debug(f"  - language: {self.language}")
        logger.debug(f"  - sample_rate: {self.sample_rate}")
        logger.debug(f"  - channels: {self.channels}")

    @abstractmethod
    async def recognize(self, audio: AudioData) -> str:
        """识别音频，返回文本"""
        raise NotImplementedError("ASR 子类必须实现 recognize 方法")

    async def process_audio(
        self,
        audio: AudioData,
        session_id: str
    ) -> TextData:
        """处理音频并构建 TextData，内部调用 recognize"""
        if not self.is_ready:
            raise ModuleProcessingError(f"ASR 模块 {self.module_id} 未就绪")

        try:
            text = await self.recognize(audio)

            if text:
                logger.info(f"ASR [{self.module_id}] (Session: {session_id}) 识别成功: '{text[:50]}...'")
            else:
                logger.info(f"ASR [{self.module_id}] (Session: {session_id}) 未识别到文本")

            return TextData(
                text=text or "",
                chunk_id=session_id,
                is_final=True,
                language=self.language,
                metadata={
                    "source_module_id": self.module_id,
                    "sample_rate": audio.sample_rate,
                }
            )

        except Exception as e:
            logger.error(
                f"ASR [{self.module_id}] (Session: {session_id}) 识别失败: {e}",
                exc_info=True
            )
            return TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                language=self.language,
                metadata={
                    "error": str(e),
                    "source_module_id": self.module_id
                }
            )

    async def process_audio_stream(
        self,
        audio_stream: AsyncGenerator[AudioData, None],
        session_id: str
    ) -> AsyncGenerator[TextData, None]:
        """流式处理音频，默认实现逐块调用 process_audio"""
        logger.info(f"ASR [{self.module_id}] (Session: {session_id}) 流式处理开始")

        try:
            async for audio_chunk in audio_stream:
                if audio_chunk and audio_chunk.data:
                    result = await self.process_audio(audio_chunk, session_id)
                    yield result

            logger.info(f"ASR [{self.module_id}] (Session: {session_id}) 流式处理结束")

        except Exception as e:
            logger.error(
                f"ASR [{self.module_id}] (Session: {session_id}) 流式处理失败: {e}",
                exc_info=True
            )
            yield TextData(
                text="",
                chunk_id=session_id,
                is_final=True,
                language=self.language,
                metadata={
                    "error": f"stream_error: {e}",
                    "source_module_id": self.module_id
                }
            )
