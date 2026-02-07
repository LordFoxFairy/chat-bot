from typing import AsyncGenerator, Optional, Dict, Any, Type

from core.exceptions import ModuleInitializationError, ModuleProcessingError
from models import AudioData, TextData, AudioFormat
from modules.base_tts import BaseTTS
from utils.logging_setup import logger

# 动态导入 edge-tts
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    logger.warning("edge-tts 库未安装，EdgeTTSAdapter 将无法使用。请运行 'pip install edge-tts'")
    edge_tts = None  # type: ignore
    EDGE_TTS_AVAILABLE = False


class EdgeTTSAdapter(BaseTTS):
    """Edge TTS 语音合成适配器

    使用 Microsoft Edge TTS 服务进行语音合成。
    """

    # 默认配置常量
    DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
    DEFAULT_RATE = "+0%"
    DEFAULT_VOLUME = "+0%"
    DEFAULT_PITCH = "+0Hz"

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ):
        super().__init__(module_id, config)

        if not EDGE_TTS_AVAILABLE:
            raise ModuleInitializationError("edge-tts 库未安装")

        # 读取 EdgeTTS 特定配置
        self.rate = self.config.get("rate", self.DEFAULT_RATE)
        self.volume = self.config.get("volume", self.DEFAULT_VOLUME)
        self.pitch = self.config.get("pitch", self.DEFAULT_PITCH)
        self.output_format = AudioFormat.MP3  # EdgeTTS 输出 MP3

        logger.info(f"TTS/EdgeTTS [{self.module_id}] 配置加载完成:")
        logger.info(f"  - voice: {self.voice}")
        logger.info(f"  - rate: {self.rate}")
        logger.info(f"  - volume: {self.volume}")
        logger.info(f"  - pitch: {self.pitch}")

    async def _setup_impl(self):
        """初始化 Edge TTS (内部实现)"""
        logger.info(f"TTS/EdgeTTS [{self.module_id}] 正在初始化...")

        try:
            # 测试连接（带超时控制）
            async with asyncio.timeout(10.0):  # 10秒超时
                communicate = edge_tts.Communicate("测试", self.voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio" and chunk["data"]:
                        logger.info(f"TTS/EdgeTTS [{self.module_id}] 连接测试成功")
                        break

            logger.info(f"TTS/EdgeTTS [{self.module_id}] 初始化成功")

        except asyncio.TimeoutError:
            logger.error(f"TTS/EdgeTTS [{self.module_id}] 初始化超时")
            raise ModuleInitializationError("EdgeTTS 初始化超时")
        except Exception as e:
            logger.error(f"TTS/EdgeTTS [{self.module_id}] 初始化失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"EdgeTTS 初始化失败: {e}") from e

    async def close(self):
        """关闭 TTS 适配器"""
        logger.info(f"TTS/EdgeTTS [{self.module_id}] 正在关闭...")
        logger.info(f"TTS/EdgeTTS [{self.module_id}] 已关闭")
        await super().close()

    async def synthesize_stream(self, text: TextData) -> AsyncGenerator[AudioData, None]:
        """流式合成语音"""
        if not text.text or not text.text.strip():
            logger.debug(f"TTS/EdgeTTS [{self.module_id}] 文本为空")
            yield AudioData(
                data=b"",
                format=self.output_format,
                is_final=True,
                metadata={"status": "empty_input"}
            )
            return

        try:
            logger.debug(f"TTS/EdgeTTS [{self.module_id}] 开始合成")

            communicate = edge_tts.Communicate(
                text.text,
                self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )

            chunk_index = 0
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes = chunk["data"]
                    if audio_bytes:
                        logger.debug(
                            f"TTS/EdgeTTS [{self.module_id}] 生成音频块 {chunk_index}: {len(audio_bytes)} 字节"
                        )
                        yield AudioData(
                            data=audio_bytes,
                            format=self.output_format,
                            is_final=False,
                            metadata={"chunk_index": chunk_index}
                        )
                        chunk_index += 1

            logger.debug(f"TTS/EdgeTTS [{self.module_id}] 合成结束，共 {chunk_index} 个音频块")

            # 发送最终标记
            yield AudioData(
                data=b"",
                format=self.output_format,
                is_final=True,
                metadata={"status": "complete", "total_chunks": chunk_index}
            )

        except Exception as e:
            logger.error(f"TTS/EdgeTTS [{self.module_id}] 合成失败: {e}", exc_info=True)
            raise ModuleProcessingError(f"合成失败: {e}") from e


def load() -> Type[BaseTTS]:
    """加载 EdgeTTS 适配器类"""
    return EdgeTTSAdapter
