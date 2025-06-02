# chat-bot/adapters/tts/edge_tts_adapter.py
import asyncio
import logging
import uuid
from typing import AsyncGenerator, Optional, Dict, Any

from core.exceptions import ModuleInitializationError  # 假設的異常類
from data_models.audio_data import AudioData, AudioFormat
from data_models.text_data import TextData
from modules.base_tts import BaseTTS  # 導入修正後的 BaseTTS

logger = logging.getLogger(__name__)

_edge_tts_available = False
edge_tts = None  # type: ignore
try:
    import edge_tts  # type: ignore

    _edge_tts_available = True
    logger.info("edge-tts 庫已成功導入。")
except ImportError:  # pragma: no cover
    logger.error("'edge-tts' 庫未安裝或無法導入。EdgeTTSAdapter 將不可用。")
    logger.error("請運行 'pip install edge-tts' 來安裝。")
except Exception as e:  # pragma: no cover
    logger.error(f"導入 edge-tts 時發生未知錯誤: {e}")


class EdgeTTSAdapter(BaseTTS):
    """
    使用 Microsoft Edge TTS 服務的適配器。
    實現 BaseTTS 接口，提供原生音頻流。
    """

    def __init__(self, module_id: Optional[str] = "edge_tts",
                 module_name: Optional[str] = "edge_tts_adapter",
                 config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 event_manager: Optional[Any] = None):

        super().__init__(module_id, module_name, config, event_loop, event_manager)

        if not _edge_tts_available:
            raise ModuleInitializationError(
                f"EdgeTTSAdapter [{self.module_id}]: 'edge-tts' 庫未安裝或導入失敗。"
            )
        self.tts_voice: str = self.adapter_specific_config.get("voice", "zh-CN-XiaoxiaoNeural")
        self.tts_rate: str = self.adapter_specific_config.get("rate", "+0%")
        self.tts_volume: str = self.adapter_specific_config.get("volume", "+0%")
        self.tts_pitch: str = self.adapter_specific_config.get("pitch", "+0Hz")

        # EdgeTTS 原生輸出特性
        # EdgeTTS 通常輸出特定格式的 MP3，例如 audio-24khz-48kbitrate-mono-mp3
        output_audio_format: str = self.adapter_specific_config.get("output_audio_format", "mp3").lower()
        try:
            self.output_audio_format_enum: AudioFormat = AudioFormat[output_audio_format.upper()]
        except KeyError:
            logger.error(f"EdgeTTSAdapter 配置的 native_format '{output_audio_format}' 無效。將默認為 MP3。")
            self.output_audio_format_enum = AudioFormat.MP3

        # 這些是 EdgeTTS MP3 流常見的參數
        self.sample_rate: int = self.adapter_specific_config.get("sample_rate", 24000)
        self.channels: int = self.adapter_specific_config.get("channels", 1)
        # 對於 MP3 等壓縮格式，sample_width 指的是解碼後 PCM 的 sample_width。
        # pydub (如果用於後續處理) 通常將 MP3 解碼為 16-bit PCM，所以 sample_width 為 2。
        # 如果適配器直接輸出 PCM，則此值應為 PCM 的實際 sample_width。
        # 由於 EdgeTTS 直接輸出 MP3，這個值更多的是一個“預期解碼後”的值。
        self.sample_width: int = self.adapter_specific_config.get("sample_width", 2)

        logger.info(f"EdgeTTSAdapter [{self.module_id}] 初始化。")
        logger.info(
            f"  TTS 引擎參數: voice='{self.tts_voice}', rate='{self.tts_rate}', volume='{self.tts_volume}', pitch='{self.tts_pitch}'")
        logger.info(f"  原生輸出特性: format={self.output_audio_format_enum.value}, sample_rate={self.sample_rate}Hz, "
                    f"channels={self.channels}, (預期解碼後) sample_width={self.sample_width}bytes")

    async def initialize(self):
        """異步初始化檢查，例如測試與TTS服務的連接。"""
        if not _edge_tts_available or edge_tts is None:
            logger.error(f"EdgeTTSAdapter [{self.module_id}] 初始化失敗: edge-tts 庫不可用。")
            return

        try:
            communicate = edge_tts.Communicate("测试", self.tts_voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and chunk["data"]:
                    logger.info(f"EdgeTTSAdapter [{self.module_id}] 連接測試成功，聲音: {self.tts_voice}")
                    return
            logger.warning(f"EdgeTTSAdapter [{self.module_id}] 連接測試未收到音頻數據，聲音: {self.tts_voice}")
        except Exception as e:
            logger.error(f"EdgeTTSAdapter [{self.module_id}] 初始化連接測試失敗 (聲音: {self.tts_voice}): {e}",
                         exc_info=True)

    async def text_to_speech_stream(self, text_input: TextData, **kwargs: Any) -> AsyncGenerator[AudioData, None]:
        """
        將文本轉換為 EdgeTTS 的原生音頻流。
        kwargs 可以覆蓋初始化時的 tts_voice, tts_rate 等參數。
        """
        chunk_id = text_input.chunk_id if text_input.chunk_id else f"edge_tts_{uuid.uuid4().hex[:8]}"

        text_to_speak = text_input.text
        if not text_to_speak or not text_to_speak.strip():
            logger.warning(f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}): 輸入文本為空。")
            yield AudioData(data=b'', chunk_id=chunk_id,
                            format=self.output_audio_format_enum, sample_rate=self.sample_rate,
                            channels=self.channels, sample_width=self.sample_width,
                            is_final=True, metadata={"status": "empty_input"})
            return

        voice = kwargs.get("voice", self.tts_voice)
        rate = kwargs.get("rate", self.tts_rate)
        volume = kwargs.get("volume", self.tts_volume)
        pitch = kwargs.get("pitch", self.tts_pitch)

        logger.info(f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}) 開始合成: '{text_to_speak[:30]}...' "
                    f"(Voice: {voice}, Rate: {rate})")

        try:
            communicate = edge_tts.Communicate(text_to_speak, voice, rate=rate, volume=volume, pitch=pitch)
            chunk_index = 0
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes = chunk["data"]
                    if audio_bytes:
                        logger.debug(
                            f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}, 塊: {chunk_index}) 生成原生音頻 {len(audio_bytes)} 字節。")
                        yield AudioData(
                            data=audio_bytes,
                            chunk_id=chunk_id,
                            format=self.output_audio_format_enum,
                            sample_rate=self.sample_rate,
                            channels=self.channels,
                            sample_width=self.sample_width,  # 代表解碼後的 PCM sample_width
                            is_final=False,
                            metadata={"chunk_index": chunk_index,
                                      "engine_output_audio_format": self.output_audio_format_enum.value}
                        )
                        chunk_index += 1
                elif chunk["type"] == "WordBoundary":
                    pass

            logger.info(f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}) 合成流結束。")
            yield AudioData(
                data=b'', chunk_id=chunk_id,
                format=self.output_audio_format_enum, sample_rate=self.sample_rate,
                channels=self.channels, sample_width=self.sample_width,
                is_final=True, metadata={"status": "stream_end", "total_chunks": chunk_index}
            )

        except edge_tts.exceptions.NoAudioReceived:  # type: ignore
            logger.error(f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}) 未返回任何音頻數據 (NoAudioReceived)。")
            yield AudioData(data=b'', chunk_id=chunk_id,
                            format=self.output_audio_format_enum, sample_rate=self.sample_rate,
                            channels=self.channels, sample_width=self.sample_width,
                            is_final=True, metadata={"error": "no_audio_received_from_edge_tts"})
        except Exception as e:
            logger.error(f"EdgeTTSAdapter [{self.module_id}] (流ID: {chunk_id}) 語音合成時发生未預期错误: {e}",
                         exc_info=True)
            yield AudioData(data=b'', chunk_id=chunk_id,
                            format=self.output_audio_format_enum, sample_rate=self.sample_rate,
                            channels=self.channels, sample_width=self.sample_width,
                            is_final=True, metadata={"error": f"tts_synthesis_unexpected_error: {str(e)}"})
