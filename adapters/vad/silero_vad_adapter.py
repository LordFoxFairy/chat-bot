# chat-bot/adapters/vad/silero_vad_adapter.py
import torch
import numpy as np
from modules.base_vad import BaseVAD
from data_models.audio_data import AudioData
from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
import logging
import os

# 类型检查时导入 EventManager，避免循环导入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.event_manager import EventManager

logger = logging.getLogger(__name__)

class SileroVADAdapter(BaseVAD):
    """
    使用 Silero VAD 模型的 VAD 适配器。
    https://github.com/snakers4/silero-vad
    """

    def __init__(self, module_id: str,
                 config: Optional[Dict[str, Any]] = None,
                 event_manager: Optional['EventManager'] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 **kwargs):

        super().__init__(
            module_id=module_id,
            config=config,
            event_manager=event_manager,
            event_loop=event_loop
        )

        adapter_config_key = self.get_config_key()
        specific_silero_config = self.vad_processing_config.get(adapter_config_key, {})

        if not specific_silero_config:
            logger.warning(f"在配置中未找到 VAD 适配器 '{adapter_config_key}' 的特定配置。"
                                f"将使用默认的 Silero VAD 参数。请检查 module_id='{module_id}' 下的 'vad' 配置块。")

        # --- Silero VAD 特定配置 (存储以供 initialize 方法使用) ---
        self.model_repo_path = specific_silero_config.get('model_repo_path', 'snakers4/silero-vad')
        self.model_name = specific_silero_config.get('model_name', 'silero_vad')
        self.threshold = float(specific_silero_config.get('threshold', 0.5))
        self.vad_sample_rate = int(specific_silero_config.get('vad_sample_rate', 16000))
        self.min_silence_duration_ms = int(specific_silero_config.get('min_silence_duration_ms', 100))
        self.speech_pad_ms = int(specific_silero_config.get('speech_pad_ms', 30))
        self.device = specific_silero_config.get('device', 'cpu')
        self.force_reload_model = bool(specific_silero_config.get('force_reload_model', False))
        self.trust_repo_for_local = bool(specific_silero_config.get('trust_repo_for_local', True))

        # 模型和工具将在 initialize 方法中加载
        self.model = None
        self.utils = None
        self.vad_iterator = None

        # 内部状态缓冲区
        self.vad_processing_window_buffer = bytearray()
        self.current_stream_audio_buffer = bytearray()
        self.speech_start_sample_offset_in_stream = None

        logger.info(f"SileroVADAdapter (id='{self.module_id}') __init__ 完成。模型将在 initialize() 中加载。")

    async def initialize(self):
        """
        异步初始化模块，加载 Silero VAD 模型和相关工具。
        """
        await super().initialize()  # 调用父类的 initialize (如果存在)
        logger.info(f"SileroVADAdapter (id='{self.module_id}') 开始异步初始化...")
        try:
            logger.info(f"正在尝试从 '{self.model_repo_path}' 加载 Silero VAD 模型 '{self.model_name}'...")

            is_local_path = os.path.exists(self.model_repo_path) and os.path.isdir(self.model_repo_path)
            source_type = 'local' if is_local_path else 'github'

            # torch.hub.load 是一个同步操作，如果它涉及到网络下载，可能会阻塞事件循环。
            # 在实际应用中，如果模型加载非常耗时，可以考虑使用 asyncio.to_thread (Python 3.9+)
            # 或者 executor 来在单独的线程中运行它，以避免阻塞主异步流程。
            # 为简单起见，这里直接调用。
            self.model, self.utils = torch.hub.load(
                repo_or_dir=self.model_repo_path,
                model=self.model_name,
                source=source_type,
                force_reload=self.force_reload_model,
                trust_repo=self.trust_repo_for_local if is_local_path else None
            )
            self.model.to(self.device)

            (self.get_speech_timestamps_fn,
             self.save_audio_fn,
             self.read_audio_fn,
             self.VADIterator,
             self.collect_chunks_fn) = self.utils

            self.vad_iterator = self.VADIterator(
                self.model,
                threshold=self.threshold,
                sampling_rate=self.vad_sample_rate,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms
            )
            logger.info(
                f"Silero VAD 模型从 {source_type} 源 '{self.model_repo_path}' 成功加载到设备 '{self.device}'。")
            logger.info(
                f"VAD 参数: SR={self.vad_sample_rate}, Threshold={self.threshold}, Min Silence={self.min_silence_duration_ms}ms, Speech Pad={self.speech_pad_ms}ms")
            self.reset_state_internal()  # 初始化后重置一次状态

        except Exception as e:
            logger.error(f"加载 Silero VAD 模型失败: {e}", exc_info=True)
            # 根据您的错误处理策略，这里可以重新抛出异常或设置一个错误状态
            raise RuntimeError(f"Silero VAD 模型加载失败: {e}") from e
        logger.info(f"SileroVADAdapter (id='{self.module_id}') 异步初始化完成。")

    def get_config_key(self) -> str:
        return "silero_vad"

    def reset_state_internal(self):
        """内部方法，用于重置 VAD 状态和缓冲区。"""
        if self.vad_iterator:
            self.vad_iterator.reset_states()
        self.vad_processing_window_buffer.clear()
        self.current_stream_audio_buffer.clear()
        self.speech_start_sample_offset_in_stream = None
        logger.debug("SileroVADAdapter 内部状态和缓冲区已重置。")

    async def reset_state(self):
        """公共 API，用于重置 VAD 状态，由 ChatEngine 或类似模块调用。"""
        await super().reset_state()
        self.reset_state_internal()
        logger.info("SileroVADAdapter 状态已通过公共 API 显式重置。")

    async def process(self, audio_data: AudioData) -> AsyncGenerator[AudioData, None]:
        """
        使用 Silero VAD 处理传入的音频数据块。
        当检测到语音时，产生 AudioData 片段。
        """
        if not self.model or not self.vad_iterator:  # 检查模型是否已初始化
            logger.error("Silero VAD 模型未初始化。请先调用 initialize()。无法处理音频。")
            if False: yield
            return

        if audio_data.sample_rate != self.vad_sample_rate:
            logger.error(f"输入音频采样率 ({audio_data.sample_rate} Hz) 与 "
                              f"Silero VAD 期望的采样率 ({self.vad_sample_rate} Hz) 不匹配。音频将不被处理。")
            if False: yield
            return
        if audio_data.channels != 1:
            logger.warning(
                f"输入音频有 {audio_data.channels} 个声道。Silero VAD 期望单声道。将按原样使用，但结果可能不理想。")
        if audio_data.sample_width != 2:
            logger.error(f"输入音频采样宽度为 {audio_data.sample_width} 字节。Silero VAD 期望 2 字节 (16-bit PCM)。")
            if False: yield
            return

        bytes_per_sample = audio_data.sample_width
        # Silero VAD 0.10.0 之后，window_size_samples 可能不存在，或者其值可能不直接对应于 VADIterator 期望的块大小。
        # VADIterator 内部处理块大小，通常我们传递任意大小的块给它，它会自己缓冲。
        # 然而，为了演示目的，我们可以假设一个典型块大小，或者让 VADIterator 处理。
        # Silero VAD 通常在内部按 30ms (480 样本 @ 16kHz) 或类似大小的帧处理。
        # 我们传递给 VADIterator 的块大小不需要严格匹配这个，它会自己缓冲。
        # 这里我们保持之前的逻辑，即尝试按模型窗口大小（如果可用）或自定义块大小处理，
        # 但 Silero VADIterator 更灵活。

        # 尝试获取模型定义的窗口大小，如果模型对象支持
        window_size_samples = getattr(self.model, 'window_size_samples', 512)  # 默认一个常见值
        window_size_bytes = window_size_samples * bytes_per_sample

        self.current_stream_audio_buffer.extend(audio_data.audio_bytes)
        self.vad_processing_window_buffer.extend(audio_data.audio_bytes)

        # 只要缓冲区中的数据足够一个 VADIterator 的处理窗口就处理
        # 注意：VADIterator 本身会进行缓冲，所以这里的 window_size_bytes 更多是控制我们多久调用一次 VADIterator
        # 实际上，Silero 的 VADIterator 可以接受任意长度的块，它内部会处理。
        # 为了与之前的逻辑保持一致性并展示分块，我们保留这个循环。
        # 更简单的做法是直接将整个 audio_data.audio_bytes 转换为 tensor 并传递给 VADIterator。
        while len(self.vad_processing_window_buffer) >= window_size_bytes:  # 或者一个更小的、合理的处理块大小
            chunk_to_process_bytes = self.vad_processing_window_buffer[:window_size_bytes]
            self.vad_processing_window_buffer = self.vad_processing_window_buffer[window_size_bytes:]

            audio_int16 = np.frombuffer(chunk_to_process_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
            tensor_chunk = torch.from_numpy(audio_float32).to(self.device)

            speech_event = self.vad_iterator(tensor_chunk, return_seconds=False)

            if speech_event:
                if "start" in speech_event:
                    self.speech_start_sample_offset_in_stream = speech_event["start"]
                    logger.debug(
                        f"VAD: 在当前音频流的样本偏移 {self.speech_start_sample_offset_in_stream} 处检测到语音开始。")

                elif "end" in speech_event and self.speech_start_sample_offset_in_stream is not None:
                    speech_end_sample_offset_in_stream = speech_event["end"]
                    logger.debug(
                        f"VAD: 在样本偏移 {speech_end_sample_offset_in_stream} 处检测到语音结束。语音开始于 {self.speech_start_sample_offset_in_stream}。")

                    start_byte_offset = self.speech_start_sample_offset_in_stream * bytes_per_sample
                    end_byte_offset = speech_end_sample_offset_in_stream * bytes_per_sample

                    if start_byte_offset < end_byte_offset <= len(self.current_stream_audio_buffer):
                        speech_segment_bytes = self.current_stream_audio_buffer[start_byte_offset:end_byte_offset]
                        segment_timestamp = audio_data.timestamp  # 近似时间戳

                        logger.info(f"VAD: 产生语音片段。流样本偏移: "
                                         f"[{self.speech_start_sample_offset_in_stream} - {speech_end_sample_offset_in_stream}]。 "
                                         f"字节长度: {len(speech_segment_bytes)}。")
                        yield AudioData(
                            audio_bytes=speech_segment_bytes,
                            sample_rate=audio_data.sample_rate,
                            sample_width=audio_data.sample_width,
                            channels=audio_data.channels,
                            is_final=True,
                            session_id=audio_data.session_id,
                            timestamp=segment_timestamp
                        )
                    else:
                        logger.warning(f"VAD: 为语音片段提取计算了无效的字节偏移。 "
                                            f"开始样本: {self.speech_start_sample_offset_in_stream}, 结束样本: {speech_end_sample_offset_in_stream}。 "
                                            f"主缓冲区长度 (字节): {len(self.current_stream_audio_buffer)}。 "
                                            f"计算的字节偏移: [{start_byte_offset} - {end_byte_offset}]。片段已忽略。")

                    self.speech_start_sample_offset_in_stream = None

        if audio_data.is_final:
            logger.info("VAD: 收到输入音频流的 'is_final=True'。")
            # 处理缓冲区中剩余的任何数据
            if len(self.vad_processing_window_buffer) > 0:
                logger.debug(f"VAD (is_final): 处理剩余的 {len(self.vad_processing_window_buffer)} 字节数据。")
                audio_int16 = np.frombuffer(self.vad_processing_window_buffer, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / np.iinfo(np.int16).max
                tensor_chunk = torch.from_numpy(audio_float32).to(self.device)

                # 调用 VADIterator 并传入 True 来指示这是流的末尾
                speech_event = self.vad_iterator(tensor_chunk, return_seconds=False, force_flush=True)
                self.vad_processing_window_buffer.clear()

                if speech_event and "end" in speech_event and self.speech_start_sample_offset_in_stream is not None:
                    speech_end_sample_offset_in_stream = speech_event["end"]
                    logger.debug(
                        f"VAD (is_final): 在流末尾检测到语音结束于样本偏移 {speech_end_sample_offset_in_stream}。")
                    start_byte_offset = self.speech_start_sample_offset_in_stream * bytes_per_sample
                    end_byte_offset = speech_end_sample_offset_in_stream * bytes_per_sample
                    if end_byte_offset > start_byte_offset and end_byte_offset <= len(self.current_stream_audio_buffer):
                        speech_segment_bytes = self.current_stream_audio_buffer[start_byte_offset:end_byte_offset]
                        yield AudioData(
                            audio_bytes=speech_segment_bytes,
                            sample_rate=audio_data.sample_rate, sample_width=audio_data.sample_width,
                            channels=audio_data.channels, is_final=True,
                            session_id=audio_data.session_id, timestamp=audio_data.timestamp
                        )
                    self.speech_start_sample_offset_in_stream = None

            if self.speech_start_sample_offset_in_stream is not None:
                logger.warning(f"VAD: 输入流结束 (is_final=True) 时语音可能仍处于活动状态 "
                                    f"(开始偏移: {self.speech_start_sample_offset_in_stream}) 且未正常结束。 "
                                    f"此最后的语音片段可能被截断。")

            self.reset_state_internal()  # 重置所有缓冲区和 VAD 状态

    async def close(self):
        """释放 Silero VAD 模型和相关资源。"""
        await super().close()
        logger.info("正在关闭 SileroVADAdapter 并释放资源...")
        del self.model
        del self.utils
        del self.vad_iterator
        self.model = None
        self.utils = None
        self.vad_iterator = None
        if self.device == 'cuda' and torch.cuda.is_available() and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            logger.debug("已清除 CUDA 缓存。")
        logger.info("SileroVADAdapter 已关闭。")
