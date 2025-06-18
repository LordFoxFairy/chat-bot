import time
import asyncio
from collections import deque
from threading import Lock

from data_models.audio_data import AudioData, AudioFormat
from modules.base_asr import BaseASR
from modules.base_vad import BaseVAD
from core.session_context import SessionContext
from utils.logging_setup import logger


class AudioConsumer:
    """
    為單一會話處理音訊流，包含VAD、音訊緩衝、靜音偵測和ASR。
    使用 asyncio 進行全異步處理。
    """

    def __init__(self,
                 session_context: SessionContext,
                 vad_module: BaseVAD,
                 asr_module: BaseASR,
                 text_handler_callback,  # 注意：此回呼函數現在也必須是異步的 (awaitable)
                 silence_timeout: float = 2.0,
                 audio_segment_threshold: float = 0.3,
                 bytes_per_second: int = 32000):
        """
        初始化 AudioConsumer。
        Args:
            session_context (SessionContext): 當前會話的上下文。
            vad_module (BaseVAD): 語音活動偵測模組。
            asr_module (BaseASR): 語音辨識模組。
            text_handler_callback (function): 當識別完成時要呼叫的異步回呼函數。
            silence_timeout (float): 判斷為靜音的秒數閾值。
            audio_segment_threshold (float): 開始進行靜音判斷前需要累積的最短音訊秒數。
            bytes_per_second (int): 每秒的音訊位元組數 (例如 16000Hz * 16bit * 1 channel / 8 = 32000)。
        """
        self.session_context = session_context
        self.vad_module = vad_module
        self.asr_module = asr_module
        self.text_handler_callback = text_handler_callback

        # --- 配置 ---
        self._silence_timeout = silence_timeout
        self._audio_segment_threshold = audio_segment_threshold
        self._bytes_per_second = bytes_per_second

        # --- 狀態 ---
        self.audio_buffer = deque()
        self.last_speech_time = None
        self._lock = Lock()  # 對於非阻塞的同步代碼，threading.Lock 仍然可用
        self._is_processing_asr = False

        self.monitor_task: asyncio.Task = None

    def start(self):
        """啟動監控靜音的異步任務。"""
        if self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_silence())
            logger.info(f"[AudioConsumer] Async monitor started for session {self.session_context.session_id}.")

    def stop(self):
        """停止並取消監控任務。"""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
            logger.info(f"[AudioConsumer] Async monitor stopped for session {self.session_context.session_id}.")

    def process_chunk(self, chunk: bytes):
        """處理傳入的音訊區塊。"""
        with self._lock:
            if self.vad_module.is_speech_present(chunk):
                self.audio_buffer.append(chunk)
                self.last_speech_time = time.time()

    async def _monitor_silence(self):
        """在異步任務中週期性地檢查使用者是否已停止說話。"""
        try:
            while True:
                await asyncio.sleep(0.5)

                audio_to_process = None
                with self._lock:
                    if not self.audio_buffer or self._is_processing_asr or self.last_speech_time is None:
                        continue

                    buffered_data = b"".join(self.audio_buffer)
                    current_duration = len(buffered_data) / self._bytes_per_second

                    if current_duration < self._audio_segment_threshold:
                        continue

                    silence_duration = time.time() - self.last_speech_time
                    if silence_duration >= self._silence_timeout:
                        logger.info(
                            f"[AudioConsumer] Silence detected for session {self.session_context.session_id}. Preparing for ASR.")
                        self._is_processing_asr = True
                        audio_to_process = buffered_data
                        self.audio_buffer.clear()
                        self.last_speech_time = None

                if audio_to_process:
                    try:
                        await self._perform_asr(audio_to_process)
                    finally:
                        with self._lock:
                            self._is_processing_asr = False
        except asyncio.CancelledError:
            logger.info(f"Silence monitor for session {self.session_context.session_id} was cancelled.")
        except Exception as e:
            logger.error(f"Error in silence monitor for session {self.session_context.session_id}: {e}", exc_info=True)

    async def _perform_asr(self, audio_data_bytes: bytes):
        """異步地執行語音辨識，並呼叫異步的回呼函數。"""
        logger.info(f"--- [Session: {self.session_context.session_id}] Performing ASR on audio segment ---")
        try:
            audio_data = AudioData(data=audio_data_bytes, format=AudioFormat.PCM)

            # 直接 await 異步的 ASR 函數
            text_data = await self.asr_module.recognize_audio_block(audio_data, self.session_context.tag_id)

            if text_data and text_data.text.strip():
                logger.info(f"ASR result for session {self.session_context.session_id}: '{text_data.text}'")
                # Await 異步的回呼函數
                await self.text_handler_callback(text_data, {"session_id": self.session_context.session_id})
            else:
                logger.warning(f"ASR result was empty for session {self.session_context.session_id}.")

        except Exception as e:
            logger.error(f"Error during ASR for session {self.session_context.session_id}: {e}", exc_info=True)

