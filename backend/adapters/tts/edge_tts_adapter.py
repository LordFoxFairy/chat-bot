import asyncio
import os
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Type

from backend.core.models.exceptions import ModuleInitializationError, ModuleProcessingError
from backend.core.models import AudioData, TextData, AudioFormat
from backend.core.interfaces.base_tts import BaseTTS
from backend.utils.logging_setup import logger
from backend.utils.paths import resolve_project_path

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
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    def __init__(
        self,
        module_id: str,
        config: Dict[str, Any],
    ) -> None:
        super().__init__(module_id, config)

        if not EDGE_TTS_AVAILABLE:
            raise ModuleInitializationError("edge-tts 库未安装")

        # 读取 EdgeTTS 特定配置
        self.rate: str = self.config.get("rate", self.DEFAULT_RATE)
        self.volume: str = self.config.get("volume", self.DEFAULT_VOLUME)
        self.pitch: str = self.config.get("pitch", self.DEFAULT_PITCH)
        self.max_retries: int = self.config.get("max_retries", self.DEFAULT_MAX_RETRIES)
        self.retry_delay: float = self.config.get("retry_delay", self.DEFAULT_RETRY_DELAY)
        self.output_format: AudioFormat = AudioFormat.MP3  # EdgeTTS 输出 MP3

        # 音频保存配置
        self.save_audio: bool = self.config.get("save_generated_audio", False)
        self.save_path: str = str(resolve_project_path(
            self.config.get("audio_save_path", "outputs/tts_audio")
        ))

        logger.info(f"TTS/EdgeTTS [{self.module_id}] 配置加载完成:")
        logger.info(f"  - voice: {self.voice}")
        logger.info(f"  - rate: {self.rate}")
        logger.info(f"  - volume: {self.volume}")
        logger.info(f"  - pitch: {self.pitch}")
        if self.save_audio:
            logger.info(f"  - 音频保存路径: {self.save_path}")

    async def _setup_impl(self) -> None:
        """初始化 Edge TTS (内部实现)"""
        logger.info(f"TTS/EdgeTTS [{self.module_id}] 正在初始化...")

        try:
            # 创建音频保存目录
            if self.save_audio and self.save_path:
                os.makedirs(self.save_path, exist_ok=True)
                logger.info(f"TTS/EdgeTTS [{self.module_id}] 音频保存目录已创建: {self.save_path}")

            # 测试连接（带超时控制）
            # 注意：这里改为弱检查，失败不抛出异常，只记录警告
            # 这样可以在离线或网络不稳定时启动服务，依靠后续的重试机制
            try:
                async with asyncio.timeout(10.0):  # 10秒超时
                    communicate = edge_tts.Communicate("测试", self.voice)
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio" and chunk["data"]:
                            logger.info(f"TTS/EdgeTTS [{self.module_id}] 连接测试成功")
                            break
            except Exception as e:
                logger.warning(f"TTS/EdgeTTS [{self.module_id}] 连接测试失败（非致命）: {e}")
                logger.warning(f"TTS/EdgeTTS [{self.module_id}] 服务将在首次请求时重试连接")

            logger.info(f"TTS/EdgeTTS [{self.module_id}] 初始化成功")

        except Exception as e:
            # 只捕获配置相关等严重错误，网络错误已经在上面处理了
            logger.error(f"TTS/EdgeTTS [{self.module_id}] 初始化严重失败: {e}", exc_info=True)
            raise ModuleInitializationError(f"EdgeTTS 初始化严重失败: {e}") from e

    async def _close_impl(self) -> None:
        """关闭 TTS 适配器"""
        logger.info(f"TTS/EdgeTTS [{self.module_id}] 已关闭")

    async def synthesize_stream(self, text: TextData) -> AsyncGenerator[AudioData, None]:
        """流式合成语音（带重试机制）"""
        if not text.text or not text.text.strip():
            logger.debug(f"TTS/EdgeTTS [{self.module_id}] 文本为空")
            yield AudioData(
                data=b" ",  # 使用占位符数据
                format=self.output_format,
                is_final=True,
                metadata={"status": "empty_input"}
            )
            return

        retries = 0
        last_error = None

        while retries < self.max_retries:
            try:
                # 尝试执行合成
                async for chunk in self._do_synthesize(text):
                    yield chunk
                return  # 成功完成
            except Exception as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    logger.warning(
                        f"TTS/EdgeTTS [{self.module_id}] 合成失败，{self.retry_delay}秒后重试 "
                        f"({retries}/{self.max_retries}): {e}"
                    )
                    await asyncio.sleep(self.retry_delay)

        # 所有重试都失败
        logger.error(f"TTS/EdgeTTS [{self.module_id}] 合成最终失败，已重试 {self.max_retries} 次: {last_error}")
        raise ModuleProcessingError(f"合成失败 (重试耗尽): {last_error}") from last_error

    async def _do_synthesize(self, text: TextData) -> AsyncGenerator[AudioData, None]:
        """实际执行合成逻辑"""
        logger.debug(f"TTS/EdgeTTS [{self.module_id}] 开始合成")

        communicate = edge_tts.Communicate(
            text.text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )

        chunk_index = 0
        audio_buffer: list[bytes] = []  # 用于保存音频数据

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes = chunk["data"]
                if audio_bytes:
                    logger.debug(
                        f"TTS/EdgeTTS [{self.module_id}] 生成音频块 {chunk_index}: {len(audio_bytes)} 字节"
                    )
                    # 收集音频数据用于保存
                    if self.save_audio:
                        audio_buffer.append(audio_bytes)

                    yield AudioData(
                        data=audio_bytes,
                        format=self.output_format,
                        is_final=False,
                        metadata={"chunk_index": chunk_index}
                    )
                    chunk_index += 1

        logger.debug(f"TTS/EdgeTTS [{self.module_id}] 合成结束，共 {chunk_index} 个音频块")

        # 保存音频文件
        saved_path: str | None = None
        if self.save_audio and audio_buffer:
            saved_path = self._save_audio_file(audio_buffer, text.text)

        # 发送最终标记
        yield AudioData(
            data=b" ",  # 占位符
            format=self.output_format,
            is_final=True,
            metadata={
                "status": "complete",
                "total_chunks": chunk_index,
                "saved_path": saved_path
            }
        )

    def _save_audio_file(self, audio_chunks: list[bytes], text: str) -> str | None:
        """保存音频文件到指定目录

        Args:
            audio_chunks: 音频数据块列表
            text: 原始文本（用于生成文件名）

        Returns:
            保存的文件路径，失败返回 None
        """
        try:
            # 生成文件名: 时间戳_UUID_文本前缀.mp3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            short_id = str(uuid.uuid4())[:8]
            # 取文本前10个字符作为文件名一部分，过滤特殊字符
            text_prefix = "".join(c for c in text[:10] if c.isalnum() or c in "_ ")
            text_prefix = text_prefix.strip().replace(" ", "_") or "audio"

            filename = f"{timestamp}_{short_id}_{text_prefix}.mp3"
            filepath = os.path.join(self.save_path, filename)

            # 合并并写入文件
            audio_data = b"".join(audio_chunks)
            with open(filepath, "wb") as f:
                f.write(audio_data)

            logger.info(f"TTS/EdgeTTS [{self.module_id}] 音频已保存: {filepath} ({len(audio_data)} 字节)")
            return filepath

        except Exception as e:
            logger.error(f"TTS/EdgeTTS [{self.module_id}] 保存音频失败: {e}", exc_info=True)
            return None


def load() -> Type[BaseTTS]:
    """加载 EdgeTTS 适配器类"""
    return EdgeTTSAdapter
