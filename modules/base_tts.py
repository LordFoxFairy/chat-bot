import asyncio
import os
import uuid
from abc import abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any, Union  # 新增 Union

from data_models.audio_data import AudioData, AudioFormat
from data_models.text_data import TextData
from modules.base_module import BaseModule
from utils.logging_setup import logger


# AudioConverter 相關代碼已移除

class BaseTTS(BaseModule):  # 假設 BaseModule 是正確的基類
    """
    異步文本轉語音 (TTS) 模塊的基類。
    負責加載頂層 TTS 配置，並根據 enable_module 提取適配器特定配置。
    適配器將輸出其原生音頻格式，BaseTTS 負責保存（如果啟用）和轉發。
    """

    def __init__(self, module_id: str,
                 module_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 event_loop: Optional[asyncio.AbstractEventLoop] = None, ):

        super().__init__(module_id, config, event_loop)

        # 優先使用 'enable_module'，如果不存在則嘗試 'adapter_type'
        # 如果 YAML 中 'adapter_type' 在頂層，而 'enable_module' 在 'config' 內部，這裡的邏輯可能需要調整
        # 根據之前的上下文，'adapter_type' 似乎是頂層的，而 'enable_module' 可能不存在或與 adapter_type 相同
        self.enabled_adapter_name: Optional[str] = self.config.get("adapter_type", self.config.get("enable_module"))
        # 通用輸出設置僅保留與保存相關的配置
        self.save_generated_audio: bool = self.config.get('save_generated_audio', False)
        self.audio_save_path: str = self.config.get('audio_save_path', "outputs/tts_audio/")

        self.adapter_specific_config: Dict[str, Any] = {}
        if self.enabled_adapter_name:
            # 假設適配器特定配置位於 self.config['config'][self.enabled_adapter_name]
            # 或者如果 'config' 本身就是適配器配置 (如果只有一個適配器定義在模塊級別)
            # 這裡的邏輯基於 self.config 是模塊的完整配置塊
            all_adapter_configs_container = self.config.get("config", {})  # YAML中的 'config:' 下的內容
            if isinstance(all_adapter_configs_container, dict):
                self.adapter_specific_config = all_adapter_configs_container.get(self.enabled_adapter_name,
                                                                                 self.config)  # 如果頂層 config 就是適配器配置
            else:
                logger.warning(f"TTS 模塊 [{self.module_id}] 的 'config' 字段不是一個有效的字典。")

            if not self.adapter_specific_config and self.config.get(self.enabled_adapter_name):  # 檢查頂層是否有適配器名作為鍵
                self.adapter_specific_config = self.config.get(self.enabled_adapter_name, {})
            elif not self.adapter_specific_config:  # 如果在 'config' 下找不到，並且頂層也不是，則可能直接在 self.config
                self.adapter_specific_config = self.config  # 將整個模塊配置視為適配器配置

            if not self.adapter_specific_config:  # 最後的檢查
                logger.warning(
                    f"在 TTS 模塊 [{self.module_id}] 下未找到適配器 '{self.enabled_adapter_name}' 的特定配置。將使用根配置。")
                self.adapter_specific_config = self.config  # 再次確保有值

        else:
            logger.warning(f"TTS 模塊 [{self.module_id}] 配置中未指定 'enable_module' 或 'adapter_type'。將使用根配置。")
            self.adapter_specific_config = self.config  # 如果沒有指定適配器，則使用整個模塊配置

        if self.save_generated_audio:  # 移除了 os.path.exists 檢查，將在保存時創建
            logger.info(f"音頻將保存在: {self.audio_save_path} (如果目錄不存在，將在首次保存時創建)")

        logger.info(f"BaseTTS [{self.module_id}] 初始化。適配器: {self.enabled_adapter_name or '未指定'}")
        logger.info(f"  保存設置: save_audio={self.save_generated_audio}, save_path='{self.audio_save_path}'")
        logger.debug(f"  適配器特定配置將使用: {self.adapter_specific_config}")

    @abstractmethod
    async def text_to_speech_stream(self, text_input: TextData, **kwargs: Any) -> AsyncGenerator[AudioData, None]:
        """
        【子類實現】核心方法：將文本轉換為音頻流。
        適配器應產生 AudioData 對象，其中包含 TTS 引擎的 *原生* 音頻塊及其格式信息
        (AudioFormat 枚舉, sample_rate, channels, sample_width)。
        """
        if False:  # pragma: no cover
            yield AudioData(data=b"", format=AudioFormat.PCM,
                            is_final=True, metadata={})

    async def _save_audio_segment(self, audio_data_to_save: AudioData, segment_identifier: Union[int, str]) -> None:
        """
        異步保存音頻片段或完整音頻。
        如果 segment_identifier 是 "complete"，則保存為完整文件。
        """
        if not self.save_generated_audio or not audio_data_to_save.data:
            return

        file_extension = audio_data_to_save.format.value.lower()

        if isinstance(segment_identifier, str) and segment_identifier == "complete":
            # 使用 chunk_id 和 _complete 後綴命名完整文件
            filename = f"{audio_data_to_save.chunk_id}_complete.{file_extension}"
        elif isinstance(segment_identifier, int):
            # 用於分段保存（如果將來需要）
            filename = f"{audio_data_to_save.chunk_id}_seg_{segment_identifier}.{file_extension}"
        else:
            logger.warning(
                f"TTS [{self.module_id}]: 無效的 segment_identifier '{segment_identifier}' 用於保存音頻。跳過保存。")
            return

        filepath = os.path.join(self.audio_save_path, filename)

        try:
            # 在保存前確保目錄存在
            if not os.path.exists(self.audio_save_path):
                os.makedirs(self.audio_save_path, exist_ok=True)
                logger.info(f"TTS [{self.module_id}]: 音頻保存路徑已創建: {self.audio_save_path}")

            with open(filepath, "wb") as f:
                f.write(audio_data_to_save.data)
            logger.info(f"TTS [{self.module_id}]: 已保存音頻: {filepath} (格式: {file_extension})")
        except Exception as e:
            logger.error(f"TTS [{self.module_id}]: 保存音頻 {filepath} 失敗: {e}", exc_info=True)

    async def process_stream(self, text_input: TextData, **kwargs: Any) -> AsyncGenerator[AudioData, None]:
        """
        處理來自適配器的音頻流，進行保存（如果啟用），並直接轉發適配器的原生音頻數據。
        音頻數據將在流結束後一次性保存。
        """
        chunk_id = text_input.chunk_id if text_input.chunk_id else f"tts_stream_{uuid.uuid4().hex[:8]}"
        logger.info(f"TTS ProcessStream [{self.module_id}] 開始處理流: {chunk_id}, 文本: '{text_input.text[:30]}...'")

        accumulated_audio_bytes = bytearray()
        first_valid_audio_chunk_props: Optional[Dict[str, Any]] = None

        try:
            async for native_audio_data in self.text_to_speech_stream(text_input, **kwargs):
                current_audio_data = AudioData(
                    data=native_audio_data.data,
                    format=native_audio_data.format,
                    is_final=native_audio_data.is_final,
                    metadata={**(native_audio_data.metadata or {}), "source_module_id": self.module_id}
                )

                if current_audio_data.data:
                    accumulated_audio_bytes.extend(current_audio_data.data)
                    if first_valid_audio_chunk_props is None:  # 捕獲第一個有效塊的屬性
                        first_valid_audio_chunk_props = {
                            "format": current_audio_data.format,
                            "sample_rate": current_audio_data.sample_rate,
                            "channels": current_audio_data.channels,
                            "sample_width": current_audio_data.sample_width,
                        }

                yield current_audio_data  # 轉發每個塊

                if current_audio_data.is_final:
                    logger.info(
                        f"TTS ProcessStream [{self.module_id}] (流ID: {chunk_id}) 收到適配器的 is_final=True 標記。")
                    # is_final 標記表示適配器流結束，此後不應再有數據塊。
                    # 我們在此處 break，然後在循環外處理保存。
                    break

            # 檢查是否需要保存以及是否有數據和屬性
            if self.save_generated_audio and accumulated_audio_bytes and first_valid_audio_chunk_props:
                logger.info(f"TTS ProcessStream [{self.module_id}] (流ID: {chunk_id}) 準備保存累積的音頻數據。")
                complete_audio_to_save = AudioData(
                    data=bytes(accumulated_audio_bytes),
                    format=first_valid_audio_chunk_props["format"],
                    is_final=True,  # 這是完整的音頻
                    metadata={"source_module_id": self.module_id, "status": "complete_audio_saved_after_stream"}
                )
                # 使用 "complete" 標識符來指示保存完整文件
                await self._save_audio_segment(complete_audio_to_save, "complete")
            elif self.save_generated_audio and not accumulated_audio_bytes:
                logger.info(f"TTS ProcessStream [{self.module_id}] (流ID: {chunk_id}) 無累積音頻數據可保存。")

            logger.info(f"TTS ProcessStream [{self.module_id}] 完成處理流: {chunk_id}")

        except Exception as e:
            logger.error(f"TTS ProcessStream [{self.module_id}] 處理過程中發生錯誤 (流ID: {chunk_id}): {e}",
                         exc_info=True)
            # 發生錯誤時，產生一個錯誤標記的 AudioData
            yield AudioData(
                data=b"",
                format=AudioFormat.PCM,
                is_final=True,
                metadata={"error": f"tts_processing_error: {str(e)}", "source_module_id": self.module_id}
            )

    async def text_to_speech_block(self, text_input: TextData, **kwargs: Any) -> Optional[AudioData]:
        """
        將文本轉換为單個完整的音頻數據塊。
        這是對 process_stream 的一個便捷封裝，音頻將是適配器的原生格式。
        """
        all_audio_bytes = bytearray()
        final_metadata = {"source_module_id": self.module_id}
        aggregated_data_shell: Optional[AudioData] = None  # 用於存儲第一個塊的元數據作為基礎

        # process_stream 內部會處理 chunk_id
        async for audio_data_chunk in self.process_stream(text_input, **kwargs):
            if aggregated_data_shell is None and audio_data_chunk.data:  # 捕獲第一個非空塊的屬性
                aggregated_data_shell = AudioData(
                    data=b'',  # 初始為空，稍後填充
                    format=audio_data_chunk.format,
                    is_final=True,  # 最終結果總是 is_final
                    metadata=final_metadata  # 初始元數據
                )

            if audio_data_chunk.data:
                all_audio_bytes.extend(audio_data_chunk.data)

            if audio_data_chunk.metadata:  # 合併所有塊的元數據（儘管通常只有最後一個重要）
                final_metadata.update(audio_data_chunk.metadata)

            if audio_data_chunk.is_final:
                # 如果 aggregated_data_shell 仍然是 None (例如，流是空的但有 is_final)
                if aggregated_data_shell is None:
                    aggregated_data_shell = AudioData(
                        data=b'', format=audio_data_chunk.format, is_final=True, metadata=final_metadata
                    )
                break  # is_final 標誌著流的結束

        if aggregated_data_shell is not None:
            aggregated_data_shell.data = bytes(all_audio_bytes)
            aggregated_data_shell.metadata = final_metadata  # 確保使用最新的元數據
            logger.info(
                f"TTS [{self.module_id}] text_to_speech_block 生成了 {len(aggregated_data_shell.data)} 字節的音頻。")
            return aggregated_data_shell

        logger.warning(
            f"TTS [{self.module_id}] text_to_speech_block 未能生成任何音頻數據 (TextData StreamID: {text_input.chunk_id})。")
        return None

    async def run(self, input_data: TextData, **kwargs: Any) -> AsyncGenerator[AudioData, None]:
        """運行 TTS 模塊的入口方法，默認使用流式處理。"""
        async for audio_data in self.process_stream(input_data, **kwargs):
            yield audio_data
