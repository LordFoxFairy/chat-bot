import time
import asyncio
from collections import deque
from threading import Lock
import uuid
import re  # 导入正则表达式模块

from data_models import StreamEvent, EventType
from data_models.audio_data import AudioData, AudioFormat
from data_models.text_data import TextData
from modules.base_asr import BaseASR
from modules.base_vad import BaseVAD
from core.session_context import SessionContext
from utils.logging_setup import logger


class AudioConsumer:
    """
    为单一会话处理音频流。
    每个实例为一个独立用户服务，在其自己的 asyncio.Task 中运行监控循环，
    这种模式可以高效地扩展到大量并发用户。
    """

    def __init__(self,
                 session_context: SessionContext,
                 vad_module: BaseVAD,
                 asr_module: BaseASR,
                 asr_result_callback,
                 silence_timeout: float = 1.0,
                 max_buffer_duration: float = 5.0,
                 audio_segment_threshold: float = 0.3,
                 bytes_per_second: int = 32000):
        self.session_context = session_context
        self.vad_module = vad_module
        self.asr_module = asr_module
        self.asr_result_callback = asr_result_callback

        self._silence_timeout = silence_timeout
        self._max_buffer_duration = max_buffer_duration
        self._audio_segment_threshold = audio_segment_threshold
        self._bytes_per_second = bytes_per_second

        self.audio_buffer = deque()
        self.last_speech_time = None
        self._lock = Lock()
        self._is_processing_asr = False
        self.full_transcript_this_turn = ""
        self.monitor_task: asyncio.Task = None

        self.client_speech_ended = asyncio.Event()

    def start(self):
        """为当前用户启动独立的监控任务。"""
        if self.monitor_task is None:
            self.monitor_task = asyncio.create_task(self._monitor_audio())
            logger.info(f"[AudioConsumer] Async monitor started for session {self.session_context.session_id}.")

    def stop(self):
        """停止并取消当前用户的监控任务。"""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None
            logger.info(f"[AudioConsumer] Async monitor stopped for session {self.session_context.session_id}.")

    def process_chunk(self, chunk: bytes):
        """处理传入的音频区块，使用VAD过滤无效音频。"""
        with self._lock:
            if self.vad_module.is_speech_present(chunk):
                # 只有包含语音的音频块才会被处理
                self.audio_buffer.append(chunk)
                self.last_speech_time = time.time()

    def signal_client_speech_end(self):
        """由外部调用，用于设置前端VAD发送的结束信号。"""
        logger.info(f"Received client speech end signal for session {self.session_context.session_id}")
        self.client_speech_ended.set()

    async def _monitor_audio(self):
        """
        在独立的异步任务中运行，持续监控音频缓冲区。
        """
        try:
            while True:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(asyncio.sleep(0.2)),
                        asyncio.create_task(self.client_speech_ended.wait())
                    ],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()

                audio_to_process = None
                is_final = False

                with self._lock:
                    if not self.audio_buffer or self._is_processing_asr:
                        if self.client_speech_ended.is_set():
                            self.client_speech_ended.clear()
                        continue

                    buffered_data = b"".join(self.audio_buffer)
                    current_duration = len(buffered_data) / self._bytes_per_second

                    if self.client_speech_ended.is_set():
                        logger.info(
                            f"Client VAD ended. Processing final segment for session {self.session_context.session_id}.")
                        is_final = True
                    elif self.last_speech_time and (time.time() - self.last_speech_time >= self._silence_timeout) and (
                            current_duration >= self._audio_segment_threshold):
                        logger.info(
                            f"Backend VAD timeout ({self._silence_timeout}s). Processing final segment for session {self.session_context.session_id}.")
                        is_final = True
                    elif current_duration >= self._max_buffer_duration:
                        logger.info(
                            f"Max buffer duration reached. Processing intermediate segment for session {self.session_context.session_id}.")
                        is_final = False

                    if is_final or current_duration >= self._max_buffer_duration:
                        audio_to_process = buffered_data
                        self.audio_buffer.clear()
                        if is_final:
                            self.last_speech_time = None
                            self.client_speech_ended.clear()

                if audio_to_process:
                    try:
                        with self._lock:
                            self._is_processing_asr = True
                        await self._perform_asr(audio_to_process, is_final)
                    finally:
                        with self._lock:
                            self._is_processing_asr = False
        except asyncio.CancelledError:
            logger.info(f"Audio monitor for session {self.session_context.session_id} was cancelled.")
        except Exception as e:
            logger.error(f"Error in audio monitor for session {self.session_context.session_id}: {e}", exc_info=True)

    async def _perform_asr(self, audio_data_bytes: bytes, is_final: bool):
        logger.info(
            f"--- [Session: {self.session_context.session_id}] Performing ASR on segment (is_final: {is_final}) ---")

        # 新增：在处理整个音频段前，再进行一次VAD检查
        if not self.vad_module.is_speech_present(audio_data_bytes):
            logger.warning(
                f"Final audio segment for session {self.session_context.session_id} discarded by VAD. No speech detected.")
            # 如果这是最后一个片段（即使是静音），也需要发送一个空的回调，以确保上游逻辑（如上下文重置）能够被触发
            if is_final:
                await self.asr_result_callback(
                    StreamEvent(
                        event_type=EventType.ASR_RESULT,
                        event_data=TextData(text="", is_final=True),
                        session_id=self.session_context.session_id
                    ),
                    {"session_id": self.session_context.session_id}
                )
            return

        try:
            audio_data = AudioData(data=audio_data_bytes, format=AudioFormat.PCM)
            text_data_segment = await self.asr_module.recognize_audio_block(audio_data, self.session_context.tag_id)

            if text_data_segment and text_data_segment.text:
                # 新增：使用正则表达式清洗ASR返回的文本，移除特殊标记
                cleaned_text = re.sub(r'<\|.*?\|>', '', text_data_segment.text).strip()

                if cleaned_text:
                    logger.info(f"ASR raw: '{text_data_segment.text}' -> Cleaned: '{cleaned_text}'")
                    self.full_transcript_this_turn += cleaned_text + " "

            if is_final:
                final_text = self.full_transcript_this_turn.strip()
                logger.info(f"Final transcript for session {self.session_context.session_id}: '{final_text}'")

                callback_text_data = TextData(
                    text=final_text,
                    is_final=True,
                    message_id=str(uuid.uuid4())
                )
                event = StreamEvent(
                    event_type=EventType.ASR_RESULT,
                    event_data=callback_text_data,
                    session_id=self.session_context.session_id,
                    tag_id=self.session_context.tag_id
                )
                await self.asr_result_callback(event, {"session_id": self.session_context.session_id})

                self.full_transcript_this_turn = ""

        except Exception as e:
            logger.error(f"Error during ASR for session {self.session_context.session_id}: {e}", exc_info=True)

