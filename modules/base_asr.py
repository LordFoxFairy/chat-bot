from abc import abstractmethod
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator, TYPE_CHECKING, List

import numpy as np

from core_framework.exceptions import ModuleProcessingError  # 確保導入 ModuleProcessingError
# 假設 BaseModule 從您的項目路徑導入
from modules.base_module import BaseModule
# 導入框架定義的核心數據模型
from data_models import AudioData, TextData, AudioFormat
# 導入音頻轉換工具
from utils.audio_converter import convert_to_target_format  # 確保路徑正確

if TYPE_CHECKING:
    from core_framework.event_manager import EventManager

# 日誌記錄器
import logging

logger = logging.getLogger(__name__)


class BaseASR(BaseModule):
    """
    異步語音識別 (ASR) 模塊的基類。
    負責加載頂層 ASR 配置，並根據 enable_module 提取適配器特定配置。
    提供通用的音頻處理流程、流式控制邏輯以及 TextData 的構建。
    子類僅需實現核心的 _infer_and_parse_audio_numpy 方法，該方法僅返回識別文本。
    """

    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None,
                 event_manager: Optional['EventManager'] = None):
        # config 參數此處預期是 YAML 中 modules.asr 下的整個字典
        super().__init__(module_id, config, event_loop, event_manager)  # self.config 將是頂層 ASR 配置

        # 從頂層 ASR 配置 (self.config) 中讀取通用設置
        self.enabled: bool = self.config.get("enabled", False)
        self.module_category: str = self.config.get("module_category", "asr")

        # enable_module 決定了要加載哪個適配器的特定配置
        # 優先使用 "enable_module"，如果不存在則回退到 "adapter_type"
        self.enabled_adapter_name: str = self.config.get("enable_module", self.config.get("adapter_type", ""))

        if not self.enabled_adapter_name:
            logger.warning(
                f"ASR 模塊 [{self.module_id}] 配置中未找到 'enable_module' 或 'adapter_type'，無法確定要加載的適配器配置。")
            self.adapter_specific_config: Dict[str, Any] = {}
        else:
            logger.info(f"ASR 模塊 [{self.module_id}] 啟用的適配器名稱: {self.enabled_adapter_name}")

            # 從頂層配置的 "config" 子字典中，根據 enabled_adapter_name 提取特定適配器的配置
            all_adapter_configs = self.config.get("config", {})
            if not isinstance(all_adapter_configs, dict):
                logger.warning(
                    f"ASR 模塊 [{self.module_id}] 的頂層 'config' 字段不是一個有效的字典，無法加載適配器特定配置。")
                self.adapter_specific_config: Dict[str, Any] = {}
            else:
                self.adapter_specific_config: Dict[str, Any] = all_adapter_configs.get(self.enabled_adapter_name, {})
                if not self.adapter_specific_config:
                    logger.warning(
                        f"在 ASR 模塊 [{self.module_id}] 的 'config' 下未找到適配器 '{self.enabled_adapter_name}' 的特定配置。將使用空配置。")

        # 從提取出的 adapter_specific_config 中加載核心音頻參數
        # 這些鍵名應與您 YAML 中為特定適配器定義的鍵名一致
        self.language: str = self.adapter_specific_config.get("language", "zh-CN")  # YAML key: language
        self.expected_sample_rate: int = self.adapter_specific_config.get("sample_rate", 16000)  # YAML key: sample_rate
        self.expected_channels: int = self.adapter_specific_config.get("channels", 1)  # YAML key: channels
        self.expected_sample_width: int = self.adapter_specific_config.get("sample_width", 2)  # YAML key: sample_width

        # BaseASR 的通用配置（也可以由 adapter_specific_config 覆蓋）
        self.expected_audio_format_for_conversion: str = self.adapter_specific_config.get(
            "expected_audio_format_for_conversion", "pcm")
        self.target_numpy_format: str = self.adapter_specific_config.get("target_numpy_format", "pcm_f32le")
        self.asr_engine_name: str = self.adapter_specific_config.get("asr_engine_name",
                                                                     self.enabled_adapter_name or self.module_id)

        logger.debug(f"BaseASR [{self.module_id}] 初始化完成。")
        logger.debug(f"  - 全局 ASR enabled: {self.enabled}")
        logger.debug(f"  - 全局 ASR module_category: {self.module_category}")
        logger.debug(f"  - 啟用的適配器名稱 (enabled_adapter_name): {self.enabled_adapter_name}")
        logger.debug(f"  - 適配器特定配置 ({self.enabled_adapter_name}) 加載的參數:")
        logger.debug(f"    - language: {self.language}")
        logger.debug(f"    - expected_sample_rate: {self.expected_sample_rate}")
        logger.debug(f"    - expected_channels: {self.expected_channels}")
        logger.debug(f"    - expected_sample_width: {self.expected_sample_width}")
        logger.debug(f"    - expected_audio_format_for_conversion: {self.expected_audio_format_for_conversion}")
        logger.debug(f"    - target_numpy_format: {self.target_numpy_format}")
        logger.debug(f"    - asr_engine_name: {self.asr_engine_name}")

    @abstractmethod
    async def _infer_and_parse_audio_numpy(self, audio_np: np.ndarray, sample_rate: int) -> Optional[str]:
        """
        【子類實現】核心方法：執行 ASR 模型推理並返回識別出的文本字符串。

        此方法僅負責從音頻數據中獲取純文本結果。

        參數:
            audio_np (np.ndarray): 經過預處理的 NumPy 音頻數組。
            sample_rate (int): 音頻數據的採樣率。

        返回:
            Optional[str]: 識別出的文本字符串。如果模型調用失敗或未識別到文本，可以返回 None 或空字符串。
        """
        raise NotImplementedError("ASR 子類必須實現 _infer_and_parse_audio_numpy 方法，該方法僅返回文本字符串。")

    async def _recognize_audio_data(self, audio_input: AudioData, session_id: str, is_final_input: bool) -> Optional[
        TextData]:
        """
        受保護的內部方法，封裝了從 AudioData 到 TextData 的核心識別流程。
        """
        log_prefix = f"BaseASR [{self.module_id}] (Session: {session_id})"
        common_metadata = {"asr_engine": self.asr_engine_name, "source_module_id": self.module_id}

        if not audio_input.data:
            logger.debug(f"{log_prefix} 接收到空的音頻數據。")
            if is_final_input:
                return TextData(text="", chunk_id=session_id, is_final=True, language=self.language,
                                metadata={**common_metadata, "status": "empty_input_audio_data"})
            return None

        audio_np = convert_to_target_format(
            audio_input=audio_input,
            target_sample_rate=self.expected_sample_rate,
            target_channels=self.expected_channels,
            target_sample_width=self.expected_sample_width,
            target_format_for_asr=self.target_numpy_format
        )

        if audio_np is None or audio_np.size == 0:
            logger.warning(f"{log_prefix} 音頻數據轉換後為空或轉換失敗。")
            if is_final_input:
                return TextData(text="", chunk_id=session_id, is_final=True, language=self.language,
                                metadata={**common_metadata, "status": "audio_conversion_failed_or_empty"})
            return None

        recognized_text_str: Optional[str] = None  # 變量名修改以反映其類型
        try:
            # _infer_and_parse_audio_numpy 現在只返回 Optional[str]
            recognized_text_str = await self._infer_and_parse_audio_numpy(audio_np, self.expected_sample_rate)
        except Exception as e:
            logger.error(f"{log_prefix} 調用 _infer_and_parse_audio_numpy 時發生錯誤: {e}", exc_info=True)
            if is_final_input:
                return TextData(text="", chunk_id=session_id, is_final=True, language=self.language,
                                metadata={**common_metadata, "error": f"infer_and_parse_error: {e}"})
            return None

        # 構造 TextData 對象
        # 如果 recognized_text_str 為 None (表示推理失敗或無結果)，則視為空文本
        final_text_to_use = recognized_text_str if recognized_text_str is not None else ""

        if not final_text_to_use.strip() and not is_final_input:
            # 中間結果且文本為空，則不產生 TextData
            logger.debug(f"{log_prefix} 中間結果解析文本為空，不生成 TextData。")
            return None

        # 其他情況 (is_final_input=True，或者 is_final_input=False 但文本非空) 都應產生 TextData
        return TextData(
            text=final_text_to_use,
            chunk_id=session_id,  # session_id 在此處填充
            is_final=is_final_input,
            language=self.language,
            confidence=None,  # 因為 _infer_and_parse_audio_numpy 只返回文本，置信度設為 None
            metadata=common_metadata  # 元數據只包含通用部分，不再嘗試合併 parsed_info.get("metadata")
        )

    async def recognize_audio_block(self, audio_input: AudioData, session_id: str) -> Optional[TextData]:
        log_prefix = f"BaseASR [{self.module_id}] (Session: {session_id}) [recognize_audio_block]"
        logger.debug(
            f"{log_prefix} 接收到 AudioData (格式: {audio_input.format.value if audio_input.format else 'N/A'}, 採樣率: {audio_input.sample_rate} Hz)")

        if not self.is_ready:
            logger.error(f"{log_prefix} 模塊未初始化或未就緒。")
            raise ModuleProcessingError(f"ASR模塊 {self.module_id} 未就緒。")

        recognized_text_data = await self._recognize_audio_data(audio_input, session_id, is_final_input=True)

        if recognized_text_data is None:
            logger.error(
                f"{log_prefix} _recognize_audio_data 在 is_final_input=True 時返回了 None，這不符合預期。創建默認空 TextData。")
            return TextData(text="", chunk_id=session_id, is_final=True, language=self.language,
                            metadata={"status": "block_processing_unexpected_none_from_core",
                                      "source_module_id": self.module_id, "asr_engine": self.asr_engine_name})

        if recognized_text_data.text:
            logger.info(f"{log_prefix} ASR成功: '{recognized_text_data.text[:50]}...'")
        else:
            logger.info(f"{log_prefix} ASR完成，但未識別到文本或結果為空。")

        return recognized_text_data

    async def stream_recognize_audio(self,
                                     audio_chunk_stream: AsyncGenerator[AudioData, None],
                                     session_id: str) -> AsyncGenerator[TextData, None]:
        log_prefix = f"BaseASR [{self.module_id}] (Session: {session_id}) [stream_recognize_audio]"
        logger.info(f"{log_prefix} 流式 ASR 已啟動。")

        if not self.is_ready:
            logger.error(f"{log_prefix} 模塊未初始化或未就緒。")
            yield TextData(text="", chunk_id=session_id, is_final=True,
                           metadata={"error": "module_not_ready", "source_module_id": self.module_id,
                                     "asr_engine": self.asr_engine_name})
            return

        continuous_audio_buffer: List[bytes] = []
        stream_processing_threshold_bytes = self.adapter_specific_config.get(
            "stream_processing_threshold_bytes",
            self.expected_sample_rate * self.expected_sample_width * self.expected_channels * 1
        )
        last_yielded_intermediate_text = ""
        chunk_idle_timeout = self.adapter_specific_config.get("chunk_idle_timeout_ms", 500) / 1000.0
        last_valid_audio_chunk_for_metadata: Optional[AudioData] = None
        final_result_yielded_flag = False

        try:
            while True:
                audio_chunk: Optional[AudioData] = None
                try:
                    audio_chunk = await asyncio.wait_for(audio_chunk_stream.__anext__(), timeout=chunk_idle_timeout)
                    if audio_chunk and audio_chunk.data:
                        last_valid_audio_chunk_for_metadata = audio_chunk
                except StopAsyncIteration:
                    logger.info(f"{log_prefix} 音頻輸入流已正常結束。")
                    break
                except asyncio.TimeoutError:
                    logger.debug(f"{log_prefix} 等待音頻塊超時 ({chunk_idle_timeout}s)。")
                    if not continuous_audio_buffer:
                        continue
                    logger.info(f"{log_prefix} 音頻流空閒超時，處理當前緩存數據作為一個片段。")

                if audio_chunk and audio_chunk.data:
                    continuous_audio_buffer.append(audio_chunk.data)

                current_buffer_size = sum(len(b) for b in continuous_audio_buffer)
                should_process_buffer = (current_buffer_size >= stream_processing_threshold_bytes) or \
                                        (audio_chunk is None and continuous_audio_buffer)

                if should_process_buffer:
                    combined_audio_bytes = b"".join(continuous_audio_buffer)
                    continuous_audio_buffer.clear()

                    logger.debug(f"{log_prefix} 緩衝區達到條件，處理片段，長度: {len(combined_audio_bytes)} 字節。")

                    current_chunk_for_metadata = audio_chunk if audio_chunk else last_valid_audio_chunk_for_metadata

                    try:
                        chunk_format_str = self.expected_audio_format_for_conversion
                        if current_chunk_for_metadata:
                            chunk_format_str = current_chunk_for_metadata.format.value
                        chunk_format = AudioFormat(chunk_format_str)
                    except (ValueError, KeyError):
                        logger.warning(f"{log_prefix} 無效的音頻格式字符串: {chunk_format_str}。回退到 PCM。")
                        chunk_format = AudioFormat.PCM

                    chunk_sr = current_chunk_for_metadata.sample_rate if current_chunk_for_metadata else self.expected_sample_rate
                    chunk_ch = current_chunk_for_metadata.channels if current_chunk_for_metadata else self.expected_channels
                    chunk_sw = current_chunk_for_metadata.sample_width if current_chunk_for_metadata else self.expected_sample_width

                    temp_audio_data = AudioData(
                        data=combined_audio_bytes,
                        format=chunk_format, sample_rate=chunk_sr,
                        channels=chunk_ch, sample_width=chunk_sw
                    )

                    intermediate_text_data = await self._recognize_audio_data(temp_audio_data, session_id,
                                                                              is_final_input=False)

                    if intermediate_text_data and intermediate_text_data.text:
                        if intermediate_text_data.text != last_yielded_intermediate_text:
                            last_yielded_intermediate_text = intermediate_text_data.text
                            logger.debug(f"{log_prefix} 產生中間識別結果: '{last_yielded_intermediate_text[:50]}...'")
                            yield intermediate_text_data

            logger.info(f"{log_prefix} 音頻輸入流已結束。處理最終緩衝區內容。")

            if continuous_audio_buffer:
                final_audio_bytes = b"".join(continuous_audio_buffer)
                continuous_audio_buffer.clear()
                logger.debug(f"{log_prefix} 處理最終緩衝音頻，長度: {len(final_audio_bytes)} 字節。")

                current_chunk_for_metadata = last_valid_audio_chunk_for_metadata
                try:
                    chunk_format_str = self.expected_audio_format_for_conversion
                    if current_chunk_for_metadata:
                        chunk_format_str = current_chunk_for_metadata.format.value
                    chunk_format = AudioFormat(chunk_format_str)
                except (ValueError, KeyError):
                    chunk_format = AudioFormat.PCM

                chunk_sr = current_chunk_for_metadata.sample_rate if current_chunk_for_metadata else self.expected_sample_rate
                chunk_ch = current_chunk_for_metadata.channels if current_chunk_for_metadata else self.expected_channels
                chunk_sw = current_chunk_for_metadata.sample_width if current_chunk_for_metadata else self.expected_sample_width

                final_temp_audio_data = AudioData(
                    data=final_audio_bytes, format=chunk_format,
                    sample_rate=chunk_sr, channels=chunk_ch, sample_width=chunk_sw
                )

                final_text_data_obj = await self._recognize_audio_data(final_temp_audio_data, session_id,
                                                                       is_final_input=True)

                if final_text_data_obj:
                    logger.info(f"{log_prefix} 產生最終識別結果: '{final_text_data_obj.text[:50]}...'")
                    yield final_text_data_obj
                    final_result_yielded_flag = True

            elif last_yielded_intermediate_text:
                logger.info(f"{log_prefix} 音頻流結束時緩衝區已為空，將累計的中間文本作為最終結果。")
                yield TextData(text=last_yielded_intermediate_text, chunk_id=session_id, is_final=True,
                               language=self.language,
                               metadata={"status": "finalizing_last_intermediate", "source_module_id": self.module_id,
                                         "asr_engine": self.asr_engine_name})
                final_result_yielded_flag = True

            if not final_result_yielded_flag:
                logger.info(f"{log_prefix} 音頻流結束時無任何識別文本，產生空的最終結果。")
                yield TextData(text="", chunk_id=session_id, is_final=True, language=self.language,
                               metadata={"status": "empty_stream_final_result", "source_module_id": self.module_id,
                                         "asr_engine": self.asr_engine_name})

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} 流式 ASR 任務被取消。")
            raise
        except Exception as e:
            logger.error(f"{log_prefix} 流式 ASR 處理过程中发生意外错误: {e}", exc_info=True)
            yield TextData(text="", chunk_id=session_id, is_final=True,
                           metadata={"error": f"stream_processing_error: {e}", "source_module_id": self.module_id,
                                     "asr_engine": self.asr_engine_name})
        finally:
            continuous_audio_buffer.clear()
            logger.info(f"{log_prefix} 流式 ASR 已結束。")
